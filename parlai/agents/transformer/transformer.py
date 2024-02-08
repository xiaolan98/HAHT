#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer Agents.
"""
import os.path
from typing import List, Optional, Tuple
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import recursive_getattr, warn_once
from parlai.utils.logging import logging
from parlai.core.torch_agent import Output

from .modules import (
    TransformerMemNetModel,
    TransformerGeneratorModel,
    TransformerLinearWrapper,
)

import torch
from parlai.core.metrics import AverageMetric
from parlai.core.torch_generator_agent import PPLMetric


def add_common_cmdline_args(parser):
    """
    Add common command line args.
    """
    parser.add_argument(
        '-esz',
        '--embedding-size',
        type=int,
        default=300,
        help='Size of all embedding layers. Must be a multiple of --n-heads.',
    )
    parser.add_argument(
        '-nl', '--n-layers', type=int, default=2, help='Number of transformer layers.'
    )
    parser.add_argument(
        '-hid',
        '--ffn-size',
        type=int,
        default=300,
        help='Hidden size of the FFN layers',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout used around embeddings and before layer layer normalizations. '
        'This is used in Vaswani 2017 and works well on large datasets.',
    )
    parser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax. This is not used in Vaswani 2017.',
    )
    parser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after the ReLU in the FFN. Not used in Vaswani 2017, '
        'but used in Tensor2Tensor.',
    )
    parser.add_argument(
        '--n-heads', type=int, default=2, help='Number of multihead attention heads'
    )
    parser.add_argument(
        '--learn-positional-embeddings',
        type='bool',
        default=False,
        help='If off, sinusoidal embeddings are used. If on, position embeddings are '
        'learned from scratch.',
    )
    parser.add_argument('--embeddings-scale', type='bool', default=True)
    parser.add_argument(
        '--n-positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    parser.add_argument(
        '--n-segments',
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
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    parser.add_argument(
        '--share-word-embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for candidate and context'
        'in the memory network',
    )
    parser.add_argument(
        '-nel',
        '--n-encoder-layers',
        type=int,
        default=-1,
        help='This will overidde the n-layers for asymmetrical transformers',
    )
    parser.add_argument(
        '-ndl',
        '--n-decoder-layers',
        type=int,
        default=-1,
        help='This will overidde the n-layers for asymmetrical transformers',
    )
    parser.add_argument(
        '--model-parallel',
        type='bool',
        default=False,
        help='Shard the layers across multiple GPUs.',
    )
    parser.add_argument(
        '--checkpoint-activations',
        type='bool',
        default=False,
        help='Recompute activations on backward pass to conserve memory.',
    )
    parser.add_argument(
        '--log_path',
        type=str,
        default="./logs/",
        help='logs to save generated responses',
    )
    parser.add_argument(
        '--enlarge_position_embeddings',
        type=bool,
        default=False,
        help='whether to enlarge the position embeddings. ',
    )


class Transformer(Agent):
    """
    Placeholder Transformer Agent.

    Placeholder class, which just throws an error telling the user to specify whether
    they want the ranker or the generator.
    """

    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model transformer` is not a valid choice. Please select either "
            "`--model transformer/ranker` or `--model transformer/generator"
        )


class TransformerRankerAgent(TorchRankerAgent):
    """
    Transformer Ranker Agent.

    Implementation of a TorchRankerAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument(
            '--use-memories',
            type='bool',
            default=False,
            help='use memories: must implement the function '
            '`_vectorize_memories` to use this',
        )
        agent.add_argument(
            '--wrap-memory-encoder',
            type='bool',
            default=False,
            help='wrap memory encoder with MLP',
        )
        agent.add_argument(
            '--memory-attention',
            type=str,
            default='sqrt',
            choices=['cosine', 'dot', 'sqrt'],
            help='similarity for basic attention mechanism '
            'when using transformer to encode memories',
        )
        # model specific arguments
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--share-encoders', type='bool', default=True)
        parser.add_argument(
            '--share-word-embeddings',
            type='bool',
            default=True,
            help='Share word embeddings table for candidate and context'
            'in the memory network',
        )
        agent.add_argument(
            '--learn-embeddings', type='bool', default=True, help='learn embeddings'
        )
        agent.add_argument(
            '--data-parallel',
            type='bool',
            default=False,
            help='use model in data parallel, requires ' 'multiple gpus',
        )
        agent.add_argument(
            '--reduction-type',
            type=str,
            default='mean',
            choices=['first', 'max', 'mean'],
            help='Type of reduction at the end of transformer',
        )

        parser.set_defaults(learningrate=0.0001, optimizer='adamax', truncate=1024)
        cls.dictionary_class().add_cmdline_args(parser, partial_opt=partial_opt)

        return agent

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1), cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError(
                'Unexpected candidate dimensions {}' ''.format(cands.dim())
            )

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerMemNetModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model

    def batchify(self, obs_batch, sort=False):
        """
        Override so that we can add memories to the Batch object.
        """
        batch = super().batchify(obs_batch, sort)
        if self.opt['use_memories']:
            valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]
            valid_inds, exs = zip(*valid_obs)
            mems = None
            if any('memory_vecs' in ex for ex in exs):
                mems = [ex.get('memory_vecs', None) for ex in exs]
            batch.memory_vecs = mems
        return batch

    def _vectorize_memories(self, obs):
        # TODO: move this to Torch Ranker Agent
        raise NotImplementedError(
            'Abstract class: user must implement this function to use memories'
        )

    def vectorize(self, *args, **kwargs):
        """
        Override to include vectorization of memories.
        """
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        obs = super().vectorize(*args, **kwargs)
        if self.opt['use_memories']:
            obs = self._vectorize_memories(obs)
        return obs

    def encode_candidates(self, padded_cands):
        """
        Encode candidates.
        """
        _, cands = self.model(xs=None, mems=None, cands=padded_cands)

        return cands

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Score candidates.
        """
        # convoluted check that not all memories are empty
        if (
            self.opt['use_memories']
            and batch.memory_vecs is not None
            and sum(len(m) for m in batch.memory_vecs)
        ):
            mems = padded_3d(batch.memory_vecs, pad_idx=self.NULL_IDX)
        else:
            mems = None

        if cand_encs is not None:
            # we pre-encoded the candidates, do not re-encode here
            cand_vecs = None

        context_h, cands_h = self.model(xs=batch.text_vec, mems=mems, cands=cand_vecs)

        if cand_encs is not None:
            cands_h = cand_encs
        scores = self._score(context_h, cands_h)

        return scores


class TransformerGeneratorAgent(TorchGeneratorAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(parser, partial_opt=partial_opt)

        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

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

    def __init__(self, opt, shared=None):
        self.valid_output = []
        self.valid_input = []
        self.valid_ground_truth = []
        self.valid_history = []
        self.log_path = opt.get("log_path", "./log/")
        self.add_user_token = opt.get("add_user_token", False)
        self.enlarge_position_embeddings = opt.get("enlarge_position_embeddings", False)
        super().__init__(opt, shared=shared)

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

        text = [self._v2t(p) for p in preds] if preds is not None else []
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
        retval = Output(
            text, cand_choices, token_losses=token_losses, cand_scores=cand_scores
        )
        if not self.skip_generation:
            retval.beam_texts = beam_texts
            self.valid_output.extend(text)
            input_text = [self.v2t4truncated_txt(p).replace("__null__", "").strip() for p in
                          batch.text_vec] if batch.text_vec is not None else []
            label_text = [self.v2t4truncated_txt(p).replace("__null__", "").strip() for p in
                          batch.label_vec] if batch.label_vec is not None else []
            self.valid_input.extend(input_text)
            self.valid_ground_truth.extend(label_text)
        return retval

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        if self.add_user_token and not self.is_training:
            scores = scores[:, 2:, :].contiguous()
            label = batch.label_vec[:, 2:].contiguous()
            preds = preds[:, 2:]
        else:
            label = batch.label_vec
        score_view = scores.reshape(-1, scores.size(-1))
        loss = self.criterion(score_view, label.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = label.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((label == preds) * notnull).sum(dim=-1)

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

    def v2t4truncated_txt(self, vec):
        # change self._v2t() to the following codes
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        vec = new_vec
        # change self.dict.vec2txt(new_vec) in the self._v2t() to the following codes
        delimiter = ' '
        tokens = [self.dict[int(idx)] for idx in vec]
        if self.dict.tokenizer in ['gpt2', 'bpe', 'slow_bytelevel_bpe']:
            # if we used a BPE tokenizer we need to rejoin the encodings
            text = self.dict.bpe.decode(tokens, vec, delimiter)
        elif self.dict.tokenizer == 'bytelevelbpe':
            # We add special tokens in the beginning of ParlAI dict but in the
            # end of Hugging Face dict, there is an offset of #(extra tokens) between them.
            extra_tokens = 4  # length of special tokens
            vec = [
                self.dict.bpe.special_tok_map[int(idx)]
                if int(idx) in self.dict.bpe.special_tok_map
                else int(idx) - extra_tokens
                for idx in vec
            ]
            tokens = [self.dict[int(idx)] for idx in vec]
            # change text = self.bpe.decode(tokens, vector, delimiter) in self.dict.vec2txt(new_vec) to following codes
            if self.dict.bpe.debug:
                return delimiter.join(tokens)

            for i, token in enumerate(tokens):
                # note, HF ByteLevelBPE tokenizer handles special tokens itself in
                # a special way, so this will be skipped
                if token in self.dict.bpe._special_tokens:
                    # special token found. to the left, we've already cleared
                    left = self.dict.bpe.helper_decode(tokens[:i], vec[:i], delimiter)
                    # token itself is easy to map to a string
                    center = token
                    # to the right, there may still be special tokens
                    right = self.dict.bpe.decode(
                        tokens[min(len(vec), i + 1):],
                        vec[min(len(vec), i + 1):],
                        delimiter,
                    )
                    return left + center + right

            # no special tokens found, we can fall back
            text = self.dict.bpe.helper_decode(tokens, vec, delimiter)
            text = text.lstrip(' ')
        else:
            text = delimiter.join(self.dict[int(idx)] for idx in vec)
        return text

    def report(self):
        base = super().report()
        if not self.is_training:
            self.write_list(self.valid_output, "test_output")
            self.write_list(self.valid_input, "test_input")
            self.write_list(self.valid_ground_truth, "test_ground_truth")
        return base

    def write_list(self, output_list, name):
        file_name = self.opt["model_file"].split("/")[-2]
        if not os.path.exists(self.log_path):
            print("Create dictionary", self.log_path)
            os.mkdir(self.log_path)
        with open(self.log_path+name+"_"+file_name+".txt", "w", encoding="utf-8") as f:
            for idx, output in enumerate(output_list):
                if "history" in name:
                    for conv_idx, conv in enumerate(output):
                        f.write(str(idx)+"_"+str(conv_idx)+"_"+conv+"\n")
                else:
                    f.write(str(idx) + "_" + output + "\n")

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        if self.enlarge_position_embeddings:
            original_position_num = state_dict["encoder.position_embeddings.weight"].size(0)
            current_position_num = self.model.encoder.position_embeddings.weight.size(0)
            if original_position_num < current_position_num:
                print("current position number of larger than the pretrained one.")
                if not self.opt["learn_positional_embeddings"]:
                    state_dict["encoder.position_embeddings.weight"] = self.model.encoder.position_embeddings.weight
                    state_dict["decoder.position_embeddings.weight"] = self.model.decoder.position_embeddings.weight
                else:
                    state_dict["encoder.position_embeddings.weight"] = \
                        torch.cat(
                            [state_dict["encoder.position_embeddings.weight"],
                             self.model.encoder.position_embeddings.weight[original_position_num:].cpu()],
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
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as msg:
            msg_ = str(msg)
            if 'size mismatch' in msg_ and 'embedding' in msg_:
                if hasattr(self, 'special_toks') and len(self.special_toks) > 0:
                    state_dict = self._resize_token_embeddings(state_dict, msg_)
                    self.model.load_state_dict(state_dict)
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


class TransformerClassifierAgent(TorchClassifierAgent):
    """
    Classifier based on Transformer.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerRankerAgent.add_cmdline_args(
            parser, partial_opt=partial_opt
        )  # add transformer args
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        parser.add_argument(
            '--load-from-pretrained-ranker',
            type='bool',
            default=False,
            help='load model from base transformer ranking model '
            '(used for pretraining)',
        )
        parser.set_defaults(reduction_type='first')
        return parser

    def build_model(self):
        num_classes = len(self.class_list)
        self.base_model = TransformerMemNetModel(self.opt, self.dict)
        return TransformerLinearWrapper(self.base_model.context_encoder, num_classes)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)

        if 'text_vec' in obs and 'added_start_end' not in obs:
            obs.force_set(
                'text_vec', self._add_start_end_tokens(obs['text_vec'], True, True)
            )
            obs['added_start_end'] = True

        # check truncation after adding start end tokens
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(
                obs['text_vec'], self.text_truncate, True
            )
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))

        return obs

    def score(self, batch):
        return self.model(batch.text_vec)

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        if self.is_finetune and self.opt['load_from_pretrained_ranker']:
            self.base_model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)
