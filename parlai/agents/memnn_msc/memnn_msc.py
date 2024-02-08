#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import torch
import torch.nn.functional as F
from typing import Optional
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.agents.seq2seq.modules import opt_to_kwargs
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import warn_once, recursive_getattr
import parlai.utils.logging as logging
from typing import List, Optional, Tuple
from parlai.core.torch_agent import Output, Batch
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.torch import (
    neginf,
)

from parlai.core.metrics import AverageMetric
from parlai.core.torch_generator_agent import PPLMetric

from .generator import MemnnMsc, sequence_to_padding, reduce_output


def add_common_cmdline_args(parser):
    """
    Add common command line args.
    """
    parser.add_argument(
        '-esz',
        '--embedding_size',
        type=int,
        default=300,
        help='Size of all embedding layers. Must be a multiple of --n-heads.',
    )
    parser.add_argument(
        '-nl', '--n_layers', type=int, default=2, help='Number of transformer layers.'
    )
    parser.add_argument(
        '-hid',
        '--ffn_size',
        type=int,
        default=300,
        help='Hidden size of the FFN layers',
    )
    parser.add_argument(
        '-mem_hid',
        '--memory_hidden_size',
        type=int,
        default=300,
        help='Hidden size of the history memory',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout used around embeddings and before layer layer normalizations. '
        'This is used in Vaswani 2017 and works well on large datasets.',
    )
    parser.add_argument(
        '--attention_dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax. This is not used in Vaswani 2017.',
    )
    parser.add_argument(
        '--relu_dropout',
        type=float,
        default=0.0,
        help='Dropout used after the ReLU in the FFN. Not used in Vaswani 2017, '
        'but used in Tensor2Tensor.',
    )
    parser.add_argument(
        '--n_heads', type=int, default=2, help='Number of multihead attention heads'
    )
    parser.add_argument(
        '--learn_positional_embeddings',
        type='bool',
        default=False,
        help='If off, sinusoidal embeddings are used. If on, position embeddings are '
        'learned from scratch.',
    )
    parser.add_argument('--embeddings_scale', type='bool', default=True)
    parser.add_argument(
        '--n_positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    parser.add_argument(
        '--n_segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
        'If zero no segment and no langs_embedding.',
    )
    parser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm', 'prelayernorm', 'bart'},
        default='aiayn',
        help='Chooses locations of layer norms, etc. prelayernorm '
        'is used to match some fairseq models',
        recommended='xlm',
    )
    parser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
        'more recent papers prefer gelu.',
        recommended='gelu',
    )
    parser.add_argument(
        '--output_scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    parser.add_argument(
        '--share_word_embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for history conversation and context in the memory network',
    )
    parser.add_argument(
        '--share_word_embeddings_enc_dec',
        type='bool',
        default=True,
        help='Share word embeddings table for the encoder and decoder',
    )
    parser.add_argument(
        '-nel',
        '--n_encoder_layers',
        type=int,
        default=-1,
        help='This will overidde the n-layers for asymmetrical transformers',
    )
    parser.add_argument(
        '-ndl',
        '--n_decoder_layers',
        type=int,
        default=-1,
        help='This will overidde the n-layers for asymmetrical transformers',
    )
    parser.add_argument(
        '--model_parallel',
        type='bool',
        default=False,
        help='Shard the layers across multiple GPUs.',
    )
    parser.add_argument(
        '--checkpoint_activations',
        type='bool',
        default=False,
        help='Recompute activations on backward pass to conserve memory.',
    )
    parser.add_argument(
        '--reduction_type',
        default='mean',
        choices=['mean', 'none', 'max', 'first', None],
        help='Reduction type of the output of history conversation encoder',
    )
    parser.add_argument(
        '--num_slot',
        type=int,
        default=3,
        help="The slot number of long-term memory of history conversations"
    )
    parser.add_argument(
        "--num_hop",
        type=int,
        default=1,
        help="Number of hop, 1-hop or 2-hop"
    )
    parser.add_argument(
        "--concat_dim",
        type=int,
        default=0,
        help="The dimension to perform concatenation, "
             "0 means concatenate along the row dimension, 1 means concatenate along the column dimension"
    )
    parser.add_argument(
        "--no_retrieval_augment",
        type=bool,
        default=True,
        help="Whether to use the retrieval augment. For memnn_msc model, always set it as True"
    )
    parser.add_argument(
        "--hist_utter_separator",
        type=str,
        default="\n",
        help="the separator used to separate the sentence of conversations, "
             "previously '\n' is used, but is replaced with ' '."
    )
    parser.add_argument(
        '--msc_passage_type',
        type=str,
        default='whole',
        choices=["whole", "separate"],
        help="whole means concatenating the whole conversation into one sentence as the history_conv, "
             "separate means separating the whole conversation into utterances as the history_conv"
    )
    parser.add_argument(
        "--parallel_attention",
        type=bool,
        default=False,
        help="Whether to parallely attend to history conversation and context, "
             "inspired by the model in Evolved Transformer"
    )
    parser.add_argument(
        "--init_memory_attention",
        type=bool,
        default=False,
        help="Whether to use the encoder_decoder attention to initialize the memory attention (in the decoder)"
    )
    parser.add_argument(
        "--init_by_blender",
        type=bool,
        default=False,
        help="Whether to use Blender to initialise the memnn"
    )
    parser.add_argument(
        "--init_by_bart",
        type=bool,
        default=False,
        help="Whether to use Bart-large to initialise the memnn"
    )
    parser.add_argument(
        "--share_hist_cont_encoder",
        type=bool,
        default=False,
        help="Whether to share the history conversation encoder with context encoder"
    )
    parser.add_argument(
        "--memory_module_type",
        type=str,
        default=None,
        choices=["lstm", "transformer", "forget", "none", None],
        help="Different types of memory module;"
             "the way to process the history conversation after getting its representation from history conv encoder"
    )
    parser.add_argument(
        "--froze_ctx_encoder",
        type=bool,
        default=False,
        help="weather to keep context encoder frozen during training"
    )
    parser.add_argument(
        "--decoder_memory_attention",
        type=bool,
        default=True,
        help="whether to add a layer in the decoder to attend to the memory vector getting "
             "from history conversation encoder. "
             "If not, the memory vector will be concatenated to the context vector"
    )
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "+
             "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes."
    )
    parser.add_argument(
        '--copy_net',
        type=bool,
        default=False,
        help="Whether to adopt copyNet"
    )
    parser.add_argument(
        '--average_hist_vec',
        type=bool,
        default=True,
        help="Whether to average all the sentence vector of history conversation "
             "(applied when msc passage type is separate)"
    )
    parser.add_argument(
        '--hist_aware_cxt',
        type=bool,
        default=False,
        help="Whether to involve the history conversation representation before context encoder"
    )
    parser.add_argument(
        '--log_path',
        type=str,
        default="./logs/",
        help="Whether to involve the history conversation representation before context encoder"
    )


class MemnnMscAgent(TorchGeneratorAgent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        super().add_cmdline_args(parser, partial_opt=partial_opt)

        return parser

    def __init__(self, opt, shared=None):
        self.msc_passage_type = opt.get("msc_passage_type", "whole")
        self.init_memory_attention = opt.get("init_memory_attention", False)
        self.init_by_blender = opt.get("init_by_blender", False)
        self.init_by_bart = opt.get("init_by_bart", False)
        self.valid_output = []
        self.valid_input = []
        self.valid_ground_truth = []
        self.valid_history = []
        self.glossaries = opt.get("glossaries", [])
        self.add_user_token = opt.get("add_user_token", False)
        self.copy_net = opt.get("copy_net", False)
        self.log_path = opt.get("log_path", "./logs/")
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        super().__init__(opt, shared=shared)

    def build_criterion(self):
        """
        Construct and return the loss function.

        By default torch.nn.CrossEntropyLoss.

        If overridden, this model should produce a sum that can be used for a per-token loss.
        """
        if not self.fp16:
            if self.copy_net:
                return torch.nn.NLLLoss(
                    ignore_index=self.NULL_IDX, reduction='none'
                )
            else:
                return torch.nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, reduction='none'
                )
        else:
            # FP16 safe cross entropy (softmax done in FP32)
            if self.copy_net:
                return torch.nn.NLLLoss(
                    ignore_index=self.NULL_IDX, reduction='none'
                )
            else:
                return FP16SafeCrossEntropy(ignore_index=self.NULL_IDX, reduction='none')

    def build_dictionary(self):
        """
        Return the constructed dictionary, which will be set to self.dict.

        If you need to add additional tokens to the dictionary, this is likely the right
        place to do it.
        """
        d = self.dictionary_class()(self.opt)
        self.special_toks = self._get_special_tokens()
        if self.special_toks:
            d.add_additional_special_tokens(self.special_toks)

        if self.glossaries and self.init_by_blender:
            to_remove = ['__fp16_pad_0__', '__fp16_pad_1__', '__fp16_pad_2__', '__fp16_pad_3__']
            for token in to_remove:
                if token in d.tok2ind:
                    del d.freq[token]
                    idx = d.tok2ind.pop(token)
                    del d.ind2tok[idx]

            for tok in self.glossaries:
                d.add_token(tok)

            for i, tok in enumerate(self.glossaries):
                d.freq[tok] = 1000000000 + 4 + len(self.special_toks) + i

        if self.opt.get('person_tokens'):
            d[self.P1_TOKEN] = 999_999_999
            d[self.P2_TOKEN] = 999_999_998
        return d

    def build_model(self, states=None):
        opt = self.opt
        if not states:
            states = {}
        kwargs = opt_to_kwargs(opt)

        model = MemnnMsc(opt, self.dict)

        if states:
            # set loaded states if applicable
            model.load_state_dict(states["model"])

        if self.use_cuda:
            model.cuda()

        return model

    def batchify(self, obs_batch, sort=True):
        """
        Add action and attribute supervision for batches.

        Store history vec as context_vec.
        """
        if self.is_training:
            sort = sort
        else:
            sort = False
        batch = super().batchify(obs_batch, sort)
        # historical conversation
        # sum here is list concat, not addition
        if batch.valid_indices is None:
            return batch
        unpadded_hist_conv_vec = sum([obs_batch[i]['hist_conv_vec'] for i in batch.valid_indices], [])\
            if batch.valid_indices is not None else []
        if not unpadded_hist_conv_vec:
            hist_conv_vec = []
        else:
            hist_conv_vec, hist_conv_lens_ = self._pad_tensor(
                unpadded_hist_conv_vec
            )
        batch['hist_conv_vec'] = hist_conv_vec
        if self.msc_passage_type == "whole":
            batch['hist_conv_lens'] = torch.LongTensor(
                [len(obs_batch[i]['hist_conv_vec']) for i in batch.valid_indices]
            )
        else:
            batch['hist_conv_lens'] = torch.LongTensor(
                [len(obs_batch[i]['hist_conv_utter_len']) for i in batch.valid_indices]
            )
            batch['hist_conv_utter_lens'] = torch.LongTensor(
                sum([obs_batch[i]['hist_conv_utter_len'] for i in batch.valid_indices], [])
            )
        batch["hist_voc_mask"] = torch.BoolTensor([obs_batch[i]['hist_voc_mask'] for i in batch.valid_indices])
        return batch

    def _model_input(self, batch):
        if self.msc_passage_type == "whole":
            return batch.text_vec, batch.hist_conv_vec, batch.hist_conv_lens
        else:
            return batch.text_vec, batch.hist_conv_vec, batch.hist_conv_lens, batch.hist_conv_utter_lens

    def _set_history_conv_vec(self, obs, truncate):
        """
        Set the 'history_conv' field in the observation.

        """
        if "history_conv" not in obs:
            return obs
        if "hist_conv_vec" not in obs:
            if self.init_by_bart:
                add_start = True
            else:
                add_start = False
            add_end = True
            hist_conv_vec = [
                self._add_start_end_tokens(
                    torch.LongTensor(self._check_truncate(self.dict.txt2vec(h), truncate - 2, True)),
                    add_start,
                    add_end)
                for h in obs["history_conv"]
            ]
            hist_conv_utter_len = [
                h
                for h in obs.get("history_conv_utter_len", [])
            ]
            obs["hist_conv_vec"] = hist_conv_vec
            obs["hist_conv_utter_len"] = hist_conv_utter_len
            hist_voc_mask = [0] * len(self.dict)
            hist_voc = set(sum([h.data.tolist() for h in hist_conv_vec], []))
            for voc in hist_voc:
                hist_voc_mask[voc] = 1
            if self.add_user_token:
                if self.init_by_bart:
                    hist_voc_mask[self.dict.tok2ind["User"]] = 0
                    hist_voc_mask[self.dict.tok2ind["Assistant"]] = 0
                else:
                    hist_voc_mask[self.dict.tok2ind["user"]] = 0
                    hist_voc_mask[self.dict.tok2ind["assistant"]] = 0
            obs["hist_voc_mask"] = hist_voc_mask
            # obs["hist_voc_mask"] = torch.BoolTensor(hist_voc_mask)

        return obs

    def _set_text_vec(self, obs, history, truncate):
        """
        Override to prepend start token and append end token.
        """
        obs = super()._set_text_vec(obs, history, truncate)
        if self.init_by_bart:
            if 'text' not in obs or 'text_vec' not in obs:
                return obs
            vec = obs['text_vec']

            # add start/end tokens
            if 'added_start_end_tokens' not in obs:
                if truncate is not None:
                    vec = torch.LongTensor(  # type: ignore
                        self._check_truncate(obs['text_vec'], truncate - 2, True)
                    )
                obs.force_set(
                    'text_vec',
                    self._add_start_end_tokens(vec, add_start=True, add_end=True),
                )
                obs['added_start_end_tokens'] = True

        return obs

    def vectorize(self, obs, history, **kwargs):
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        super().vectorize(obs, history, **kwargs)
        self._set_history_conv_vec(obs, kwargs["text_truncate"])
        return obs

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        if self.init_by_blender:
            if "encoder.position_embeddings.weight" in state_dict:
                original_position_num = state_dict["encoder.position_embeddings.weight"].size(0)
                current_position_num = self.model.encoder.context_encoder.position_embeddings.weight.size(0)
                if original_position_num < current_position_num:
                    print("current position number of larger than the pretrained one.")
                    if not self.opt["learn_positional_embeddings"]:
                        state_dict["encoder.position_embeddings.weight"] = \
                            self.model.encoder.context_encoder.position_embeddings.weight
                        state_dict["decoder.position_embeddings.weight"] = \
                            self.model.decoder.position_embeddings.weight
                    else:
                        state_dict["encoder.position_embeddings.weight"] = \
                            torch.cat(
                                [state_dict["encoder.position_embeddings.weight"],
                                 self.model.encoder.context_encoder.position_embeddings.weight[original_position_num:].cpu()],
                                dim=0
                            )
                        state_dict["decoder.position_embeddings.weight"] = \
                            torch.cat(
                                [state_dict["decoder.position_embeddings.weight"],
                                 self.model.decoder.position_embeddings.weight[original_position_num:].cpu()],
                                dim=0
                            )
                elif original_position_num > current_position_num:
                    print("Too small number of position is set for the model.")

            for key in list(state_dict.keys()):
                if key.startswith('encoder'):
                    replaced_key_context = key.replace("encoder", "encoder.context_encoder")
                    replaced_key_hist = key.replace("encoder", "encoder.history_encoder")
                    state_dict[replaced_key_context] = state_dict[key]
                    state_dict[replaced_key_hist] = state_dict[key]
        elif self.init_by_bart and 'encoder.context_encoder.embeddings.weight' not in state_dict:
            state_dict_new = {}
            for key in list(state_dict.keys()):
                if key.startswith('encoder'):
                    replaced_key_context = key.replace("encoder", "encoder.context_encoder")
                    replaced_key_hist = key.replace("encoder", "encoder.history_encoder")
                    state_dict_new[replaced_key_context] = state_dict[key]
                    state_dict_new[replaced_key_hist] = state_dict[key]
                else:
                    state_dict_new[key] = state_dict[key]
            state_dict = state_dict_new

        if self.init_memory_attention:
            for key in list(state_dict.keys()):
                if key.startswith("decoder") and "memory_attention" in key:
                    replaced_key = key.replace("memory_attention", "encoder_attention")
                    state_dict[key] = state_dict[replaced_key].clone()
                elif key.startswith("decoder") and "norm4" in key:
                    replaced_key = key.replace("norm4", "norm2")
                    state_dict[key] = state_dict[replaced_key].clone()
        try:
            self.model.load_state_dict(state_dict, strict=False)
        except RuntimeError as msg:
            msg_ = str(msg)
            if 'size mismatch' in msg_ and 'embedding' in msg_:
                if hasattr(self, 'special_toks') and len(self.special_toks) > 0:
                    state_dict = self._resize_token_embeddings(state_dict, msg_)
                    self.model.load_state_dict(state_dict, strict=False)
                    self.resized_embeddings = True  # make note that we resized here
                else:
                    raise RuntimeError(
                        f'{msg_}\n'
                        '-----------------\n'
                        'Could not load the model due to a size mismatch in the '
                        'embeddings. A common reason for this is trying to load '
                        'a model trained with fp16 but loaded without fp16. Try '
                        'adding --fp16 true or --force-fp16-tokens true.'
                    )
            else:
                raise

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.
        """
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            'embeddings.weight',
            'encoder.embeddings.weight',
            'encoder.context_encoder.embeddings.weight',
            'encoder.history_encoder.embeddings.weight',
            'decoder.embeddings.weight',
        ]:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            prefix_tokens = self.get_prefix_tokens(batch)
            beam_preds_scores, beams = self._generate(
                batch, self.beam_size, maxlen, prefix_tokens=prefix_tokens
            )
            preds, scores = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            for beam in beams:
                beam_texts.append([])
                for tokens, score in beam.get_rescored_finished():
                    try:
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(batch, bsz)

        text = [self._v2t(p) for p in preds] if preds is not None else None
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
        retval = Output(
            text, cand_choices, token_losses=token_losses, cand_scores=cand_scores
        )
        if not self.skip_generation:
            if self.add_user_token:
                retval.text = [" ".join(t.split(" ")[2:]) for t in text]
            retval.beam_texts = beam_texts
            self.valid_output.extend(text)
            input_text = [self._v2t(p).replace("__null__", "").strip() for p in
                          batch.text_vec] if batch.text_vec is not None else []
            label_text = [self._v2t(p).replace("__null__", "").strip() for p in
                          batch.label_vec] if batch.label_vec is not None else []
            if self.opt["msc_passage_type"] == "separate":
                if not batch.hist_conv_vec != []:
                    hist_conv_text = [[]]*batch.batchsize
                else:
                    padded_hist_conv_vec = \
                        sequence_to_padding(
                            sequence_to_padding(batch.hist_conv_vec, lengths=batch.hist_conv_utter_lens).long(),
                            lengths=batch.hist_conv_lens).long()
                    hist_conv_text = [
                        [" ".join([self.dict.vec2txt(m).replace("__null__", "").strip() for m in k])for k in i]
                        for i in padded_hist_conv_vec.data.cpu().numpy()]
            else:
                if not batch.hist_conv_vec != []:
                    hist_conv_text = [[]]*batch.batchsize
                else:
                    if batch.hist_conv_vec is not None:
                        padded_hist_conv_vec = \
                                sequence_to_padding(batch.hist_conv_vec, lengths=batch.hist_conv_lens).long()
                        hist_conv_text = [[self._v2t(k).strip() for k in i]for i in padded_hist_conv_vec]
                    else:
                        hist_conv_text = []
            self.valid_history.extend(hist_conv_text)
            self.valid_input.extend(input_text)
            self.valid_ground_truth.extend(label_text)
        return retval

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        if self.init_by_bart:
            if len(vec) == 0 or vec[0] == 0:
                return self.dict.vec2txt(new_vec)
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def report(self):
        base = super().report()
        if not self.is_training:
            self.write_list(self.valid_output, "test_output")
            self.write_list(self.valid_input, "test_input")
            self.write_list(self.valid_ground_truth, "test_ground_truth")
            self.write_list(self.valid_history, "test_history_conversation")
        return base

    def write_list(self, output_list, name):
        file_name = self.opt["model_file"].split("/")[-2]
        save_path = os.path.join(self.log_path, name+"_"+file_name+".txt")
        with open(save_path, "w", encoding="utf-8") as f:
            for idx, output in enumerate(output_list):
                if "history" in name:
                    for conv_idx, conv in enumerate(output):
                        f.write(str(idx)+"_"+str(conv_idx)+"_"+conv+"\n")
                else:
                    f.write(str(idx) + "_" + output + "\n")

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec, hist_voc_mask=batch.hist_voc_mask)
        if self.copy_net:
            scores, hist_voc_scores, weight_score, *_ = model_output
            if hist_voc_scores is not None:
                scores = F.softmax(scores, 2, dtype=torch.float32)
                hist_voc_scores = F.softmax(hist_voc_scores, 2, dtype=torch.float32)
                seq_len = scores.size(1)
                dict_size = scores.size(2)
                scores = torch.cat([scores, hist_voc_scores], dim=2).view(-1, 2, dict_size)
                scores = torch.bmm(weight_score.view(-1, 1, 2), scores).view(-1, seq_len, dict_size)
                # scores = 0.9 * scores + 0.1 * hist_voc_scores
                scores = torch.log(scores + 1e-7)
                # scores = F.log_softmax(scores, 2, dtype=torch.float32)
                _, preds = scores.max(dim=2)
            else:
                scores = F.log_softmax(scores, 2, dtype=torch.float32)
                _, preds = scores.max(dim=2)
        else:
            scores, preds, *_ = model_output
        if self.add_user_token and not self.is_training:
            scores = scores[:, 2:, :].contiguous()
            label = batch.label_vec[:, 2:].contiguous()
            preds = preds[:, 2:]
        else:
            label = batch.label_vec
        score_view = scores.reshape(-1, scores.size(-1))
        loss = self.criterion(score_view, label.view(-1))
        # loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)

        notnull = label.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((label == preds) * notnull).sum(dim=-1)
        # notnull = batch.label_vec.ne(self.NULL_IDX)
        # target_tokens = notnull.long().sum(dim=-1)
        # correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        # cross entropy loss
        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        # perplexity
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        # token-wise accuracy
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # utterance-wise exact match
        self.record_local_metric(
            'token_em', AverageMetric.many(correct == target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        # Todo: modify the generation process to meet the copy net setting
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch).tolist()
            beams = [
                self._treesearch_factory(dev)
                .set_batch_context(batch_context_list, batch_idx)
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        if self.copy_net:
            long_time_memory = encoder_states[-1]
            if long_time_memory is not None:
                long_time_memory, *_ = reduce_output(long_time_memory[0], long_time_memory[1], self.opt["reduction_type"])
        if not self.opt.get("decoder_memory_attention", True):
            encoder_states = (encoder_states[0], encoder_states[1], None)

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break
            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            # score = score[:, -1:, :]
            # score = model.output(score)
            if self.copy_net:
                latent = score[:, -1:, :]
                scores = model.output(latent)
                scores = F.softmax(scores, 2, dtype=torch.float32)
                if long_time_memory is not None:
                    hist_voc_logits = F.linear(model.hist_conv_layer(long_time_memory), model.embeddings.weight)
                    hist_voc_logits.masked_fill_(~batch.hist_voc_mask, neginf(hist_voc_logits.dtype))
                    hist_voc_scores = hist_voc_logits.unsqueeze(1)
                    weight_score = \
                        F.softmax(model.weight_layer(torch.cat([latent, long_time_memory.unsqueeze(1)], dim=-1)).div_(2.),
                                  dim=2,
                                  dtype=torch.float32)
                    hist_voc_scores = F.softmax(hist_voc_scores, 2, dtype=torch.float32)
                    beam_size = scores.size(1)
                    dict_size = scores.size(2)
                    scores = torch.cat([scores, hist_voc_scores], dim=2).view(-1, 2, dict_size)
                    scores = torch.bmm(weight_score.view(-1, 1, 2), scores).view(-1, beam_size, dict_size)
                # scores = 0.9 * scores + 0.1 * hist_voc_scores
                score = torch.log(scores + 1e-7)

            else:
                score = score[:, -1:, :]
                score = model.output(score)
                # score contains softmax scores for bsz * beam_size samples
                score = score.view(bsz, beam_size, -1)
                if self.temperature != 1.0:
                    score.div_(self.temperature)
                # force to fp32 to avoid overflow issues during search calculations
                score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts]
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                prefix_mask[
                    :, :, prefix_toks
                ] = False  # everything except prefix toks should be neginf
                score[prefix_mask] = neginf(score.dtype)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        if self.init_by_bart:
            return (
                torch.LongTensor([self.END_IDX, self.START_IDX])  # type: ignore
                     .expand(bsz * beam_size, 2)
                     .to(dev)
            )
        else:
            return super()._get_initial_decoder_input(bsz, beam_size, dev)

