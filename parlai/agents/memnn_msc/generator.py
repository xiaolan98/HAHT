#!/usr/bin/env python3
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import torch
import torch.nn as nn
import math
import numpy as np

from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.torch import PipelineHelper
from parlai.utils.fsdp import fsdp_wrap
from parlai.nn.checkpoint import checkpoint_wrapper
from typing import Dict, Optional, Tuple
from parlai.core.torch_generator_agent import TorchGeneratorModel
import torch.nn.functional as F
from parlai.utils.torch import neginf
from .modules.decoder import MemnnMSCDecoder
from.modules.encoder import TransformerEncoder, TransformerEncoderLayer
from parlai.agents.transformer.modules import (
    create_position_codes,
    get_n_positions_from_options,
    LAYER_NORM_EPS,
    MultiHeadAttention,
    TransformerFFN,
)


def create_embeddings(dictionary, embedding_size, padding_idx):
    """
    Create and initialize word embeddings.
    """
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def sequence_to_padding(x, lengths, return_mask=False):
    """
    Return padded and reshaped sequence (x) according to tensor lengths
    Example:
        x = tensor([[1, 2], [2, 3], [4, 0], [5, 6], [7, 8], [9, 10]])
        lengths = tensor([1, 2, 2, 1])
    Would output:
        ret_tensor:
        tensor([[[1, 2], [0, 0]],
                [[2, 3], [4, 0]],
                [[5, 6], [7, 8]],
                [[9, 10], [0, 0]]])
        ret_mask:
        tensor([[1, 0],
                [1, 1],
                [1, 1],
                [1, 0]])

    """
    ret_tensor = torch.zeros(
        (lengths.shape[0], torch.max(lengths).int()) + tuple(x.shape[1:]), dtype=x.dtype
    ).to(x.device)
    ret_mask = torch.zeros(
        (lengths.shape[0], torch.max(lengths).int())
    ).to(x.device)
    cum_len = 0
    for i, l in enumerate(lengths):
        ret_tensor[i, :l] = x[cum_len: cum_len + l]
        ret_mask[i, :l] = 1
        cum_len += l
    if return_mask:
        return ret_tensor, ret_mask
    else:
        return ret_tensor


def reduce_output(input_tensor, mask, reduction_type='mean'):
    """
    Reduce transformer output at end of forward pass.

    :param input_tensor:
        encoded input
    :param mask:
        mask for encoded input
    :param reduction_type
        type of reduction

    :return (tensor, mask):
        returns the reduced tensor, and mask if appropriate
    """
    if reduction_type == 'first':
        return input_tensor[:, 0, :], None
    elif reduction_type == 'max':
        return input_tensor.max(dim=1)[0], None
    elif reduction_type == 'mean':
        divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(input_tensor)
        output = input_tensor.sum(dim=1) / divisor
        return output, None
    elif reduction_type is None or 'none' in reduction_type:
        return input_tensor, mask
    else:
        raise ValueError(
            "Can't handle --reduction-type {}".format(reduction_type)
        )


class MemnnMsc(TorchGeneratorModel):
    def __init__(
            self,
            opt,
            dictionary,
            **kwargs):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx, **kwargs)
        self.opt = opt
        self.reduction_type = opt["reduction_type"]
        if opt["share_word_embeddings_enc_dec"]:
            self.embeddings = create_embeddings(dictionary, opt["embedding_size"], padding_idx=self.pad_idx)
        else:
            self.embeddings = None
        self.encoder = MemnnMSCEncoder(opt, dictionary, embeddings=self.embeddings, padding_idx=self.pad_idx,
                                       reduction_type=self.reduction_type)
        self.decoder = MemnnMSCDecoder(opt, self.embeddings)
        self.copy_net = opt.get("copy_net", False)
        self.hist_aware_cxt = opt.get("hist_aware_cxt", False)
        self.decoder_memory_attention = opt.get("decoder_memory_attention", True)
        self.init_by_bart = opt.get("init_by_bart", False)
        if self.copy_net:
            self.weight_layer = nn.Linear(opt["embedding_size"]*2, 2)
            self.hist_conv_layer = nn.Linear(opt["embedding_size"], opt["embedding_size"])

    def reorder_encoder_states(self, encoder_states, indices):

        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        cont_vec, cont_mask, long_time_memory = encoder_states
        if not self.decoder_memory_attention:
            if long_time_memory is not None and not self.hist_aware_cxt:
                encoder_states = (torch.cat([long_time_memory[0], cont_vec], dim=1),
                                  torch.cat([long_time_memory[1], cont_mask], dim=1),
                                  long_time_memory)
        else:
            encoder_states = (cont_vec, cont_mask, long_time_memory)
        enc_output_cont, enc_cont_mask, memory_vec = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc_output_cont.device)
        enc_output_cont = torch.index_select(enc_output_cont, 0, indices)
        enc_cont_mask = torch.index_select(enc_cont_mask, 0, indices)
        if memory_vec is not None:
            memory_vec, memory_vec_mask = memory_vec
            memory_vec = torch.index_select(memory_vec, 0, indices)
            memory_vec_mask = torch.index_select(memory_vec_mask, 0, indices)
            memory_vec = (memory_vec, memory_vec_mask)
        return enc_output_cont, enc_cont_mask, memory_vec

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        if self.init_by_bart:
            # Incremental state is weird to handle when we seed decoder with two inputs initially.
            # we only have this method called when it's actually being used
            assert incremental_state is not None
            assert len(incremental_state) > 0

            for incr_state_l in incremental_state.values():
                assert 'self_attn' in incr_state_l
                assert 'prev_mask' in incr_state_l['self_attn']
                self_attn_mask = incr_state_l['self_attn']['prev_mask']
                # check this is on the very first run with incremental state
                if self_attn_mask.ndim == 3 and tuple(self_attn_mask.shape[1:]) == (2, 2):
                    # cut off the inappropriate incremental state
                    incr_state_l['self_attn']['prev_mask'] = self_attn_mask[:, -1:, :]
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        if self.init_by_bart:
            return output
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output

    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.
        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """

        if self.init_by_bart:
            # Override TGA._get_initial_forced_decoder_input to seed EOS BOS.
            tens = (
                torch.LongTensor([self.END_IDX, self.START_IDX])
                .to(inputs)
                .detach()
                .expand(bsz, 2)
            )
            return torch.cat([tens, inputs], 1)
        else:
            return super()._get_initial_forced_decoder_input(bsz, inputs)

    def decode_forced(self, encoder_states, ys, voc_mask=None):
        """
        Decode with a fixed, true sequence, computing loss.

        Useful for training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :param voc_mask:
            The hist_voc_mask is used when applying copy net

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        if self.copy_net:
            long_time_memory = encoder_states[-1]
            if long_time_memory is not None:
                long_time_memory, *_ = reduce_output(long_time_memory[0], long_time_memory[1], self.reduction_type)
        if not self.decoder_memory_attention:
            encoder_states = (encoder_states[0], encoder_states[1], None)
        latent, _ = self.decoder(inputs, encoder_states)
        if self.copy_net:
            logits = self.output(latent)
            if long_time_memory is not None:
                hist_voc_logits = F.linear(self.hist_conv_layer(long_time_memory), self.embeddings.weight)
                hist_voc_logits.masked_fill_(~voc_mask, neginf(hist_voc_logits.dtype))
                hist_voc_logits = hist_voc_logits.unsqueeze(1).repeat(1, logits.size(1), 1)
                long_time_memory = long_time_memory.unsqueeze(1).repeat(1, latent.size(1), 1)
                weight_score = \
                    F.softmax(self.weight_layer(torch.cat([latent, long_time_memory], dim=-1)).div_(2.),
                              dim=2, dtype=torch.float32)
                if self.init_by_bart and logits.size(1) != ys.size(1):
                    logits = logits[:, 1:, :]
                    hist_voc_logits = hist_voc_logits[:, 1:, :].contiguous()
                    weight_score = weight_score[:, 1:, :].contiguous()
            else:
                # print("Happens when evaluation session 1 data")
                if self.init_by_bart and logits.size(1) != ys.size(1):
                    logits = logits[:, 1:, :]
                hist_voc_logits = None
                weight_score = None
            return logits, hist_voc_logits, weight_score
        else:
            logits = self.output(latent)
            _, preds = logits.max(dim=2)
            if self.init_by_bart and logits.size(1) != ys.size(1):
                logits = logits[:, 1:, :]
                preds = preds[:, 1:]

            return logits, preds

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None, hist_voc_mask=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.
        :param hist_voc_mask:
            The hist_voc_mask is used when applying copy net

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        # use teacher forcing
        cont_vec, cont_mask, long_time_memory = encoder_states
        if not self.decoder_memory_attention:
            if long_time_memory is not None and not self.hist_aware_cxt:
                print(111)
                encoder_states = (cont_vec, cont_mask, None)
                # encoder_states = (torch.cat([long_time_memory[0], cont_vec], dim=1),
                #                   torch.cat([long_time_memory[1], cont_mask], dim=1),
                #                   long_time_memory)
        else:
            encoder_states = (cont_vec, cont_mask, long_time_memory)
        if self.copy_net:
            scores, hist_voc_logits, weight_score = self.decode_forced(encoder_states, ys, voc_mask=hist_voc_mask)
            return scores, hist_voc_logits, weight_score, encoder_states
        else:
            scores, preds = self.decode_forced(encoder_states, ys)
            return scores, preds, encoder_states


class ForgetMemModule(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.memory_hidden_size = opt['memory_hidden_size']
        self.hidden_size = opt['memory_hidden_size']
        self.num_slot = opt['num_slot']
        self.attn = MultiHeadAttention(opt, n_heads=1)
        self.mlp = nn.Sequential(nn.Linear(self.memory_hidden_size, self.memory_hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.ReLU())
        self.W = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.U = nn.Linear(self.hidden_size, self.hidden_size * 2)
        if self.hidden_size != self.memory_hidden_size:
            print("Memory hidden size should be the same as transformer hidden size")

    def forward(self, hist_conv_state, hist_conv_len, hist_conv_utter_lens=None):
        padded_hist_conv_mask = None
        if type(hist_conv_state) is not tuple:
            if hist_conv_utter_lens is None:
                padded_hist_conv_state, hist_conv_mask = \
                    sequence_to_padding(hist_conv_state, lengths=hist_conv_len, return_mask=True)
            else:
                padded_hist_conv_state1, hist_conv_utter_mask = \
                    sequence_to_padding(hist_conv_state, lengths=hist_conv_utter_lens, return_mask=True)
                padded_hist_conv_state, hist_conv_mask = \
                    sequence_to_padding(padded_hist_conv_state1, lengths=hist_conv_len, return_mask=True)
                padded_hist_conv_mask = sequence_to_padding(hist_conv_utter_mask, lengths=hist_conv_len)
        else:
            # This happens when the reduction_type='none'.
            # We will get the hidden states for all word in the history conversation
            # instead of the mean of all word representations like the situation when 'if' is true
            # TODO: cope with the case that reduction type is not mean or
            #                          that the history conversation is seperated into sentences
            padded_hist_conv_state = None
            hist_conv_mask = None
            # assume that we all will get the history conversation
        memory_init = self.init_memory(
            padded_hist_conv_state.size(0), padded_hist_conv_state.device, dtype=hist_conv_state.dtype)
        outputs = self.memory_forward(padded_hist_conv_state, memory_init, hist_conv_mask, padded_hist_conv_mask)
        long_time_memory = outputs[:, -1, :].reshape(-1, self.num_slot, self.hidden_size)
        return long_time_memory

    def init_memory(self, batch_size, device, dtype=torch.float32):
        memory = torch.stack([torch.zeros((self.num_slot, self.hidden_size), dtype=dtype)] * batch_size).to(device)
        # for _i in range(self.num_slot):
        #     nn.init.normal_(memory[:, _i], mean=_i, std=self.hidden_size ** -0.5)
        nn.init.normal_(memory, mean=0, std=self.hidden_size ** -0.5)
        return memory

    def memory_forward(self, inputs, memory, hist_conv_mask, hist_conv_utter_mask=None):
        """
        inputs: batch_size * length * hidden_size
        memory: slot_num * memory_hidden_size
        hidden_size = memory_hidden_size
        """
        memory = memory.reshape(-1, self.num_slot * self.hidden_size)
        outputs = []
        for i in range(inputs.shape[1]):
            if hist_conv_utter_mask is None:
                memory = self.memory_forward_step(inputs[:, i], memory)
            else:
                memory = self.memory_forward_step(inputs[:, i], memory, hist_conv_utter_mask[:, i])
                # memory = memory + \
                #          self.memory_forward_step(inputs[:, i], memory, hist_conv_utter_mask[:, i]) * \
                #          hist_conv_mask[:, i].unsqueeze(-1).expand(-1, memory.size(-1)).type_as(memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def memory_forward_step(self, input_vec, memory=None, mask=None):
        if memory is None:
            q = input_vec.unsqueeze(1).repeat(1, self.num_slot, 1)
            k = input_vec.unsqueeze(1).repeat(1, self.num_slot, 1)
            v = input_vec.unsqueeze(1).repeat(1, self.num_slot, 1)
            mask = torch.ones((k.size(0), k.size(1))).to(q.device) if mask is None else mask
            next_memory = self.mlp(self.attn(q, k, v, mask)[0])
            next_memory = next_memory.reshape(-1, self.num_slot * self.hidden_size)
            return next_memory
        else:
            memory = memory.reshape(-1, self.num_slot, self.memory_hidden_size)
            q = memory
            if len(memory.size()) == len(input_vec.size()):
                # k = torch.cat([memory, input_vec], 1)
                # v = torch.cat([memory, input_vec], 1)
                k = input_vec
                v = input_vec
            else:
                # print(memory.shape, input_vec.shape)
                k = torch.cat([memory, input_vec.unsqueeze(1)], 1)
                v = torch.cat([memory, input_vec.unsqueeze(1)], 1)
            mask = torch.ones((k.size(0), k.size(1))).to(q.device) if mask is None else mask
            # mask = torch.cat([torch.ones(k.shape[0], self.num_slot).to(q.device), mask], dim=1)
            # attention_debug = F.softmax(self.attn(q, k, v, mask)[2], dim=-1, dtype=torch.float).data.cpu().numpy()
            next_memory = memory + self.attn(q, k, v, mask)[0]
            # next_memory = memory + self.attn(q, k, v, mask)[0] * \
            #               mask[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, memory.size(-2), memory.size(-1))
            next_memory = next_memory + self.mlp(next_memory)
            if len(memory.size()) == len(input_vec.size()):
                gates = self.W(torch.sum(input_vec, dim=1) /
                               mask.type(input_vec.dtype).sum(dim=1).unsqueeze(1).clamp(min=1)).unsqueeze(1) +\
                        self.U(torch.tanh(memory))
            else:
                gates = self.W(input_vec.unsqueeze(1)) + self.U(torch.tanh(memory))
            gates = torch.split(gates, split_size_or_sections=self.hidden_size, dim=2)
            input_gate, forget_gate = gates
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)

            next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
            next_memory = next_memory.reshape(-1, self.num_slot * self.hidden_size)

            return next_memory


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask):
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=2)
        mask = (mask == 0)
        e.masked_fill_(mask, neginf(e.dtype))
        attention = F.softmax(e).unsqueeze(1)
        return torch.matmul(attention, h).squeeze(1)


class MemnnMSCEncoder(nn.Module):
    """
    The encoder of Memory Network for MSC, which contains one context encoder and one history conversation encoder
    """
    def __init__(self, opt, dictionary, embeddings=None, padding_idx=None, reduction_type='mean'):
        super().__init__()
        if embeddings is None:
            # pre created or trained embeddings
            if opt["share_word_embeddings"]:
                self.embeddings = create_embeddings(dictionary, opt["embedding_size"], padding_idx=padding_idx)
            else:
                self.embeddings = None
        else:
            self.embeddings = embeddings
        self._dict = dictionary
        self.hidden_size = opt['embedding_size']
        self.num_hop = opt["num_hop"]
        self.concat_dim = opt["concat_dim"]
        self.msc_passage_type = opt.get("msc_passage_type", "separate")
        self.memory_module_type = opt.get("memory_module_type", None)
        self.share_hist_cont_encoder = opt.get("share_hist_cont_encoder", False)
        self.froze_ctx_encoder = opt.get("froze_ctx_encoder", False)
        self.decoder_memory_attention = opt.get("decoder_memory_attention", True)
        self.average_hist_vec = opt.get("average_hist_vec", True)
        self.reduction_type = reduction_type
        self.new_model = opt.get("new_model", False)
        self.hist_aware_cxt = opt.get("hist_aware_cxt", False)
        if self.memory_module_type == "forget":
            self.forget_memory_module = ForgetMemModule(opt)
        elif self.memory_module_type == "transformer":
            self.attention_layer = SelfAttentionLayer(self.hidden_size, self.hidden_size)
            # self.transformer_encoder_layer = TransformerEncoderLayer(opt, n_heads=2)
            # self.n_positions = opt["n_positions"]
            # self.history_position_embeddings = nn.Embedding(self.n_positions, self.hidden_size)
            # if not opt.get('learn_positional_embeddings', False):
            #     create_position_codes(
            #         self.n_positions,
            #         self.hidden_size,
            #         out=self.history_position_embeddings.weight,
            #     )
            # else:
            #     nn.init.normal_(
            #         self.history_position_embeddings.weight, 0, self.hidden_size ** -0.5
            #     )
        elif self.memory_module_type == "user_embed":
            self.use_persona_as_history = opt.get("use_persona_as_history", False)
            self.persona_type2 = opt.get("persona_type2", "self")
        self.context_encoder = TransformerEncoder(opt, len(dictionary), self.embeddings,
                                                  padding_idx=padding_idx, reduction_type='none')
        if self.froze_ctx_encoder:
            for param in self.context_encoder.parameters():
                param.requires_grad = False
            # self.embeddings.weight.requires_grad = True
        if self.share_hist_cont_encoder:
            self.history_encoder = self.context_encoder
        else:
            self.history_encoder = TransformerEncoder(opt, len(dictionary), self.embeddings,
                                                      padding_idx=padding_idx, reduction_type=reduction_type)

    def forward(self, cont_vec, hist_conv_vec, hist_conv_len, hist_conv_utter_lens=None):
        if not hist_conv_vec != []:
            # cope with the case that there is no history conversation. The model degenerates to standard transformer
            long_time_memory = None
            cont_state = self.context_encoder(cont_vec)
            return cont_state[0], cont_state[1], long_time_memory
        else:
            hist_conv_state = self.history_encoder(hist_conv_vec)
            if self.share_hist_cont_encoder:
                hist_conv_state, *_ = reduce_output(hist_conv_state[0], hist_conv_state[1])
            if self.memory_module_type == "none" or self.memory_module_type is None:
                if self.msc_passage_type == "whole":
                    padded_hist_conv_state, hist_conv_mask = \
                        sequence_to_padding(hist_conv_state, lengths=hist_conv_len, return_mask=True)
                    long_time_memory = (padded_hist_conv_state, hist_conv_mask)
                    # long_time_memory, *_ = reduce_output(padded_hist_conv_state, hist_conv_mask)
                else:
                    assert self.msc_passage_type == "separate"
                    padded_hist_conv_state1, hist_conv_utter_mask = \
                        sequence_to_padding(hist_conv_state, lengths=hist_conv_utter_lens, return_mask=True)
                    if self.average_hist_vec:
                        padded_hist_conv_state1, *_ = \
                            reduce_output(padded_hist_conv_state1, hist_conv_utter_mask, self.reduction_type)
                        padded_hist_conv_state, hist_conv_mask = \
                            sequence_to_padding(padded_hist_conv_state1, lengths=hist_conv_len, return_mask=True)
                    else:
                        padded_hist_conv_state, *_ = \
                            sequence_to_padding(padded_hist_conv_state1, lengths=hist_conv_len, return_mask=True)
                        hist_conv_mask = sequence_to_padding(hist_conv_utter_mask, lengths=hist_conv_len)
                        batch_size = padded_hist_conv_state.size(0)
                        hidden_dim = padded_hist_conv_state.size(-1)
                        padded_hist_conv_state = padded_hist_conv_state.view(batch_size, -1, hidden_dim)
                        hist_conv_mask = hist_conv_mask.view(batch_size, -1)

                    long_time_memory = (padded_hist_conv_state, hist_conv_mask)
            elif self.memory_module_type == "transformer":
                if self.msc_passage_type == "whole":
                    padded_hist_conv_state, hist_conv_mask = \
                        sequence_to_padding(hist_conv_state, lengths=hist_conv_len, return_mask=True)
                    # seq_len = padded_hist_conv_state.size(1)
                    long_time_memory = self.attention_layer(padded_hist_conv_state, hist_conv_mask)
                    long_time_memory = (long_time_memory, hist_conv_mask)
                else:
                    assert self.msc_passage_type == "separate"
                    padded_hist_conv_state1, hist_conv_utter_mask = \
                        sequence_to_padding(hist_conv_state, lengths=hist_conv_utter_lens, return_mask=True)
                    # padded_hist_conv_state1, *_ = reduce_output(padded_hist_conv_state1, hist_conv_utter_mask)
                    # padded_hist_conv_state, hist_conv_mask = \
                    #     sequence_to_padding(padded_hist_conv_state1, lengths=hist_conv_len, return_mask=True)
                    long_time_memory = self.attention_layer(padded_hist_conv_state1, hist_conv_utter_mask)
                    long_time_memory, hist_conv_mask = \
                        sequence_to_padding(long_time_memory, lengths=hist_conv_len, return_mask=True)
                    long_time_memory = (long_time_memory, hist_conv_mask)

                # if self.msc_passage_type == "whole":
                #     padded_hist_conv_state, hist_conv_mask = \
                #         sequence_to_padding(hist_conv_state, lengths=hist_conv_len, return_mask=True)
                #     # position embedding
                #     seq_len = padded_hist_conv_state.size(1)
                #     positions = torch.arange(
                #         seq_len, dtype=torch.long, device=padded_hist_conv_state.device
                #     ).unsqueeze(0)
                #     padded_hist_conv_state += \
                #         self.history_position_embeddings(positions).expand_as(padded_hist_conv_state)
                #     long_time_memory = self.transformer_encoder_layer(padded_hist_conv_state, hist_conv_mask)
                #     long_time_memory += padded_hist_conv_state
                #     long_time_memory = (long_time_memory, hist_conv_mask)
                # else:
                #     assert self.msc_passage_type == "separate"
                #     padded_hist_conv_state1, hist_conv_utter_mask = \
                #         sequence_to_padding(hist_conv_state, lengths=hist_conv_utter_lens, return_mask=True)
                #     padded_hist_conv_state1, *_ = reduce_output(padded_hist_conv_state1, hist_conv_utter_mask)
                #     padded_hist_conv_state, hist_conv_mask = \
                #         sequence_to_padding(padded_hist_conv_state1, lengths=hist_conv_len, return_mask=True)
                #     long_time_memory = self.transformer_encoder_layer(padded_hist_conv_state, hist_conv_mask)
                #     # position embedding
                #     seq_len = long_time_memory.size(1)
                #     positions = torch.arange(
                #         seq_len, dtype=torch.long, device=long_time_memory.device
                #     ).unsqueeze(0)
                #     long_time_memory += self.history_position_embeddings(positions).expand_as(long_time_memory)
                #     long_time_memory = (long_time_memory, hist_conv_mask)
            else:
                assert self.memory_module_type == "forget", self.memory_module_type
                # long_time_memory -> (batch_size, num_slot, hidden_size)
                long_time_memory = self.forget_memory_module(hist_conv_state, hist_conv_len, hist_conv_utter_lens)
                memory_mask = torch.ones(long_time_memory.size(0), long_time_memory.size(1)).to(long_time_memory.device)
                long_time_memory = (long_time_memory, memory_mask)
            if self.hist_aware_cxt:
                cont_state = self.context_encoder(cont_vec, hist_aware_cxt_vec=long_time_memory)
                session_num = long_time_memory[0].size(1)
                long_time_memory = (cont_state[0][:, :session_num, :], long_time_memory[1])
                return cont_state[0], cont_state[1], long_time_memory
            else:
                cont_state = self.context_encoder(cont_vec)
                return cont_state[0], cont_state[1], long_time_memory
                # if not self.decoder_memory_attention:
                #     # return torch.cat([cont_state[0], long_time_memory[0]], dim=1), \
                #     #        torch.cat([cont_state[1], long_time_memory[1]], dim=1), \
                #     #        None
                #     return torch.cat([long_time_memory[0], cont_state[0]], dim=1), \
                #            torch.cat([long_time_memory[1], cont_state[1]], dim=1), \
                #            None
                # else:
                #     return cont_state[0], cont_state[1], long_time_memory



