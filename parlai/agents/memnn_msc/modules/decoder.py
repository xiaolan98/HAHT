from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from parlai.agents.transformer.modules import (
    create_position_codes,
    get_n_positions_from_options,
    LAYER_NORM_EPS,
    MultiHeadAttention,
    TransformerFFN,
)
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.torch import PipelineHelper
from parlai.utils.fsdp import fsdp_wrap
from parlai.nn.checkpoint import checkpoint_wrapper


class MemnnMSCDecoder(nn.Module):

    """
    The decoder of Memory Network for MSC. Besides attending to context states,
        decoder here will also attend to the memory of history conversation obtained from encoder

    For documentation on parameters that are take directly from opt,
    see parlai/agents/memnn_msc/modules.py

    :param opt: ParlAI-parsed options.
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        opt: Opt,
        embedding: Optional[nn.Embedding] = None,
        n_positions: Optional[int] = None,
    ):
        super().__init__()
        self.opt = opt

        def _default(val, default):
            return val if val is not None else default

        self.embedding_size = opt['embedding_size']
        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.n_heads = opt['n_heads']
        self.dim = self.embedding_size
        self.activation = opt.get('activation', 'relu')
        self.variant = opt.get('variant', 'aiayn')

        self.embeddings_scale = opt.get('embeddings_scale', True)
        self.dropout = nn.Dropout(p=opt.get('dropout', 0.0))  # --dropout

        self.n_positions = _default(n_positions, get_n_positions_from_options(opt))
        self.out_dim = self.embedding_size
        self.parallel_attention = opt.get("parallel_attention", False)
        assert (
            self.embedding_size % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
            if self.variant == 'xlm':
                warn_once(
                    'DEPRECATED: XLM should only be used for backwards compatibility, '
                    'as it involves a less-stable layernorm operation.'
                )
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.embedding_size,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size ** -0.5
            )

        # build the model
        self.layers = self.build_layers()

    def build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = TransformerDecoderLayer(
                self.opt,
                attention_dropout=self.opt.get('attention_dropout', 0.0),
                relu_dropout=self.opt.get('relu_dropout', 0.0),
                dropout=self.opt.get('dropout', 0.0),
                activation=self.activation,
                variant=self.variant,
            )
            if self.opt.get('checkpoint_activations'):
                layer = checkpoint_wrapper(layer)
            layers.append(fsdp_wrap(layer))  # type: ignore
        return layers

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch, seqlen] input:
            The target input IDs
        :param LongTensor[batch, seqlen] positions:
            Positions for input IDs. If None, computes defaults.
        :param LongTensor[batch, seqlen] segments:
            Segment IDs for extra embedding features. If None, not used.

        :return (tensor, mask):
            embedded input and mask
        """
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = self.norm_embeddings(tensor)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        return tensor

    def forward_layers(
        self,
        tensor: torch.Tensor,
        encoder_output_cont: torch.Tensor,
        encoder_cont_mask: torch.Tensor,
        memory_vec: Tuple[torch.Tensor, torch.Tensor] or None,
        incr_state: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param encoder_output_cont:
            conversation context vector
        :param encoder_cont_mask:
            conversation context mask
        :param memory_vec:
            memory vec of history conversation and mask
        :param incr_state:
            Dict mapping layer_idx to incremental state

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, encoder_output_cont, encoder_cont_mask, memory_vec, incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                tensor, new_incr_state[idx] = layer(
                    x=tensor,
                    encoder_output_cont=encoder_output_cont,
                    encoder_cont_mask=encoder_cont_mask,
                    memory_vec=memory_vec,
                    incr_state=incr_state.get(idx),
                    **kwargs,
                )

        return tensor, new_incr_state

    def forward(
        self,
        input: torch.Tensor,
        encoder_state,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output_cont, encoder_cont_mask, memory_vec = encoder_state

        seq_len = input.size(1)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=input.device
        ).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions, **kwargs)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, encoder_output_cont, encoder_cont_mask, memory_vec, incr_state, **kwargs
        )

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        return tensor, new_incr_state

    def _apply_model_parallel(self, tensor, encoder_output_cont, encoder_cont_mask, memory_vec, incr_state):
        """
        Pipeline application of model parallelism.
        """
        if memory_vec is None:
            chunks = PipelineHelper.split(
                (tensor, encoder_output_cont, encoder_cont_mask, incr_state)
            )
            work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

            new_incr_state = {i: [] for i, _ in enumerate(self.layers)}

            for chunk_idx, layer_nos, next_device in work_items:
                s_tensor, s_enc_out, s_enc_mask, s_incr_state = chunks[chunk_idx]
                for layer_no in layer_nos:
                    s_tensor, nis = self.layers[layer_no](
                        x=s_tensor,
                        encoder_output_cont=s_enc_out,
                        encoder_cont_mask=s_enc_mask,
                        memory_vec=None,
                        incr_state=s_incr_state.get(layer_no),
                    )
                    new_incr_state[layer_no].append(nis)
                # don't move incr state, it's always on the correct device
                s_tensor, s_enc_out, s_enc_mask = PipelineHelper.chunk_to(
                    (s_tensor, s_enc_out, s_enc_mask), next_device
                )
                chunks[chunk_idx] = (s_tensor, s_enc_out, s_enc_mask, s_incr_state)
        else:
            chunks = PipelineHelper.split(
                (tensor, encoder_output_cont, encoder_cont_mask, memory_vec, incr_state)
            )
            work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

            new_incr_state = {i: [] for i, _ in enumerate(self.layers)}

            for chunk_idx, layer_nos, next_device in work_items:
                s_tensor, s_enc_out_cont, s_enc_cont_mask, s_memory_vec, s_incr_state = chunks[chunk_idx]
                for layer_no in layer_nos:
                    s_tensor, nis = self.layers[layer_no](
                        x=s_tensor,
                        encoder_output_cont=s_enc_out_cont,
                        encoder_cont_mask=s_enc_cont_mask,
                        memory_vec=s_memory_vec,
                        incr_state=s_incr_state.get(layer_no),
                    )
                    new_incr_state[layer_no].append(nis)
                # don't move incr state, it's always on the correct device
                s_tensor, s_enc_out_cont, s_enc_cont_mask, s_memory_vec = PipelineHelper.chunk_to(
                    (s_tensor, s_enc_out_cont, s_enc_cont_mask, s_memory_vec), next_device
                )
                chunks[chunk_idx] = (s_tensor, s_enc_out_cont, s_enc_cont_mask, s_memory_vec, s_incr_state)

        tensor_out = PipelineHelper.join([c[0] for c in chunks])
        new_incr_state = {
            layer_no: self.join(pieces)
            for layer_no, pieces in new_incr_state.items()
        }

        return tensor_out, new_incr_state

    @classmethod
    def join(cls, items, dim=0):
        """
        Join chunks back together, the inverse of split.

        :param items:
            All the output chunks. Each chunk may be a tensor or a group of
            tensors.
        :param dim:
            The dimension to join along.
        """
        if len(items) == 0:
            raise IndexError("Cannot rejoin an empty list of chunks.")
        item0 = items[0]
        if isinstance(item0, torch.Tensor):
            # base case
            return torch.cat(items, dim=dim)  # type: ignore
        elif isinstance(item0, tuple):
            return tuple(
                cls.join(x, dim=dim) for x in zip(*items)
            )  # type: ignore
        elif isinstance(item0, dict):
            keys = item0.keys()
            return {  # type: ignore
                k: cls.join([c[k] for c in items], dim=dim)  # type: ignore
                for k in keys
            }
        elif item0 is None:
            return None
        else:
            raise TypeError(f'Cannot join list of type {type(item0)}')

    # @classmethod
    # def split(cls, item, split_size: Optional[int] = None, dim=0):
    #     """
    #     Split a tensor or group of tensors into smaller chunks of the same type.
    #
    #     :param item:
    #         The item being split. May be a Tensor, a tuple of Tensors, or a
    #         dictionary mapping str -> Tensor.
    #     :param split_size:
    #         The maximum size of each output chunk. If None, we will guess using
    #         heuristics
    #     :param dim:
    #         The dimension to split along.
    #     """
    #     if split_size is None:
    #         split_size = PipelineHelper.guess_split_size(item)
    #     if item is None:
    #         return None
    #     elif isinstance(item, torch.Tensor):
    #         # base case, just split the tensor
    #         return list(torch.split(item, split_size, dim))
    #     elif isinstance(item, tuple):
    #         # We start with Tuple[Tensor] and we return List[Tuple[Tensor]]
    #         return list(zip(*(cls.split(i, split_size, dim) for i in item)))
    #     elif isinstance(item, dict):
    #         if item == {}:
    #             # Terrible edge case: the empty dict. We handle by returning an
    #             # infinite list of empty dicts and we'll figure out its correct
    #             # size later. This happens for the incremental_state in
    #             # MultiheadAttention.
    #             return itertools.repeat({})  # type: ignore
    #
    #         # we can't handle dicts with empty objects in them, due to how we handle
    #         # the case above.  awkward syntax because pytorch 1.3 doesn't like
    #         # comparing tensors to dicts.
    #         if {} in [x for x in item.values() if isinstance(x, dict)]:
    #             raise ValueError(
    #                 'Cannot handle a dictionary with an empty dictionary inside.'
    #             )
    #         if () in [x for x in item.values() if isinstance(x, tuple)]:
    #             raise ValueError(
    #                 'Cannot handle a dictionary with an empty tuple inside.'
    #             )
    #
    #         # we start with Dict[key,tensor]
    #         # we map it to d: Dict[key, List[Tensor]], where we have split each mapping
    #         d = {k: cls.split(v, split_size, dim) for k, v in item.items()}
    #         # now we transpose it and return List[Dict[key, Tensor]]
    #         return [
    #             dict(zip(d.keys(), values))  # type: ignore
    #             for values in zip(*(d[k] for k in d.keys()))
    #         ]
    #     else:
    #         raise TypeError(f"Cannot split type {type(item)}")


class TransformerDecoderLayer(nn.Module):
    """
    Implements a single Memnn decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a causal (auto-regressive) manner.
    2. Attend over all of the encoder states.
    3. Attend over all of the memory
    """

    def __init__(
        self,
        opt: Opt,
        n_heads: int = None,
        embedding_size: int = None,
        ffn_size: int = None,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__(**kwargs)

        def _default(val, default):
            """
            shorthand for explicit None check for optional arguments.
            """
            return val if val is not None else default

        n_heads = _default(n_heads, opt['n_heads'])
        embedding_size = _default(embedding_size, opt['embedding_size'])
        ffn_size = _default(ffn_size, opt['ffn_size'])

        self.opt = opt
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            opt=self.opt, n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )  # type: ignore
        self.norm1 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            opt=self.opt, n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )  # type: ignore
        self.norm2 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.memory_attention = MultiHeadAttention(
            opt=self.opt, n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )
        self.norm4 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.parallel_attention = opt.get("parallel_attention", False)
        if self.parallel_attention:
            self.dim_align = torch.nn.Linear(embedding_size*2, embedding_size)
            nn.init.xavier_normal_(self.dim_align.weight)
        self.ffn = TransformerFFN(
            opt=self.opt,
            dim=embedding_size,
            dim_hidden=ffn_size,
            relu_dropout=relu_dropout,
            activation=activation,
        )  # type: ignore
        self.norm3 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output_cont: torch.Tensor,
        encoder_cont_mask: torch.Tensor,
        memory_vec: Tuple[torch.Tensor, torch.Tensor] or None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        The incremental state is a dict with values for self- and encoder-attention
        states.
        """

        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attention
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm1(x)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm1(x)

        if self.parallel_attention:
            if memory_vec is not None:
                memory_vec, memory_mask = memory_vec
                residual = x
                if self.variant == 'prelayernorm':
                    x = self.norm4(x)
                memory_mask = torch.ones(memory_vec.size(0), memory_vec.size(1)).to(x.device)
                x1, final_memory_attn_incr_state = self.memory_attention(
                    query=x,
                    key=memory_vec,
                    value=memory_vec,
                    mask=memory_mask,
                    incr_state=incr_state.get('memory_attn'),
                    static_kv=True,
                    **kwargs,
                )[:2]
                x1 = self.dropout(x1)  # --dropout

                # encoder_attn_layer_norm norm 2
                if self.variant == 'prelayernorm':
                    x = self.norm2(x)
                x2, final_encoder_cont_attn_incr_state = self.encoder_attention(
                    query=x,
                    key=encoder_output_cont,
                    value=encoder_output_cont,
                    mask=encoder_cont_mask,
                    incr_state=incr_state.get('encoder_cont_attn'),
                    static_kv=True,
                    **kwargs,
                )[:2]
                x2 = self.dropout(x2)  # --dropout

                x = self.dim_align(torch.cat([x1, x2], dim=-1))
                x = residual + x
                if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
                    x = self.norm2(x)
            else:
                residual = x
                # encoder_attn_layer_norm norm 2
                if self.variant == 'prelayernorm':
                    x = self.norm2(x)
                x, final_encoder_cont_attn_incr_state = self.encoder_attention(
                    query=x,
                    key=encoder_output_cont,
                    value=encoder_output_cont,
                    mask=encoder_cont_mask,
                    incr_state=incr_state.get('encoder_cont_attn'),
                    static_kv=True,
                    **kwargs,
                )[:2]
                x = self.dropout(x)  # --dropout
                x = residual + x
                if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
                    x = self.norm2(x)

                final_memory_attn_incr_state = {}
        else:
            # attention over the memory of history conversations
            if memory_vec is not None:
                memory_vec, memory_mask = memory_vec
                residual = x
                if self.variant == 'prelayernorm':
                    x = self.norm4(x)
                x, final_memory_attn_incr_state = self.memory_attention(
                    query=x,
                    key=memory_vec,
                    value=memory_vec,
                    mask=memory_mask,
                    incr_state=incr_state.get('memory_attn'),
                    static_kv=True,
                    **kwargs,
                )[:2]
                x = self.dropout(x)  # --dropout
                x = residual + x
                if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
                    x = self.norm4(x)
                _mask = memory_mask.sum(dim=1).clamp(max=1).unsqueeze(-1).unsqueeze(-1).type(x.dtype)
                x = x * _mask + residual * (1 - _mask)
            else:
                final_memory_attn_incr_state = {}

            residual = x
            # encoder_attn_layer_norm norm 2
            if self.variant == 'prelayernorm':
                x = self.norm2(x)
            x, final_encoder_cont_attn_incr_state = self.encoder_attention(
                query=x,
                key=encoder_output_cont,
                value=encoder_output_cont,
                mask=encoder_cont_mask,
                incr_state=incr_state.get('encoder_cont_attn'),
                static_kv=True,
                **kwargs,
            )[:2]
            x = self.dropout(x)  # --dropout
            x = residual + x
            if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
                x = self.norm2(x)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm3(x)
        x = self.ffn(x, **kwargs)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm3(x)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_cont_attn': final_encoder_cont_attn_incr_state,
            'memory_attn': final_memory_attn_incr_state
        }
        return x, new_incr_state

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

    def reorder_incremental_state(
        self, incremental_state: Dict[str, dict], inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'encoder_cont_attn': self.encoder_attention,
            'memory_attn': self.memory_attention
        }
        return_incremental_state = {}
        for attn_type, attn in attn_types.items():
            if incremental_state[attn_type] is not None:
                return_incremental_state[attn_type] = attn.reorder_incremental_state(incremental_state[attn_type], inds)
            else:
                return_incremental_state[attn_type] = None
        return return_incremental_state
        # return {
        #             attn_type: attn.reorder_incremental_state(
        #                 incremental_state[attn_type], inds
        #             )
        #             for attn_type, attn in attn_types.items()
        #         }