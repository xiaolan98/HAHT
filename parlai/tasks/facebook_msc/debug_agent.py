from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from .build import build
from parlai.utils.strings import normalize_reply

import json
import copy
import os


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    session_name = "session" + str(opt["session_id"])
    return os.path.join(opt['datapath'], 'facebook_msc', session_name, dt + '_convs_w_scores.txt')
    # return os.path.join(opt['datapath'], 'facebook_msc', session_name, dt + '_convs.txt')


class FacebookMSCBaseTeacher(DialogTeacher):
    @classmethod
    def add_cmdline_args(
            cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ):
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('Facebook MSC Teacher Args')
        parser.add_argument(
            '--session_id',
            '-si',
            type=int,
            default="2",
            help="session id. Session id should be from 1-5",
        )

        parser.add_argument(
            '--concat_hist_conv',
            '-ctc',
            type=bool,
            default=False,
            help="whether to concat history conversation with context",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.session_id = opt["session_id"]
        self.id = "facebook_msc" + str(self.session_id)
        self.concat_hist_conv = opt["concat_hist_conv"]
        opt['datafile'] = _path(opt)
        self.no_retrieval_augment = opt.get("no_retrieval_augment", True)
        self.msc_passage_type = opt.get("msc_passage_type", "whole")
        self.use_pre_calculated_score = opt.get("use_pre_calculated_score", False)
        self.n_docs = opt.get("n_docs", 3)
        self.hist_utter_separator = opt.get("hist_utter_separator", " ")
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        instances = []
        with open(path) as json_file:
            for line in json_file.readlines():
                instances.append(json.loads(line))
        for instance in instances:
            # user_pair = instance["user_ids"]
            history_conv = instance["history_conv"]
            current_conv = instance["current_conv"]
            your_persona = instance["your_persona"]
            partner_persona = instance["partner_persona"]
            if history_conv != []:
                history_conv = [[normalize_reply(j) for j in i] for i in history_conv]
            current_conv = [normalize_reply(i) for i in current_conv]
            if self.no_retrieval_augment:
                new_episode = True
                for sentence_id in range(1, len(current_conv), 2):
                    if self.concat_hist_conv:
                        yield {"text": " ".join([" ".join(history_i) for history_i in history_conv]) +
                                       current_conv[sentence_id - 1],
                               "label": current_conv[sentence_id],
                               "history_conv": [self.hist_utter_separator.join(history_i) for history_i in history_conv]
                               }, new_episode
                    else:
                        if self.msc_passage_type == "whole":
                            yield {"text": current_conv[sentence_id - 1], "label": current_conv[sentence_id],
                                   "history_conv": [self.hist_utter_separator.join(history_i)
                                                    for history_i in history_conv]
                                   }, new_episode
                        else:
                            assert self.msc_passage_type == "separate"
                            yield {"text": current_conv[sentence_id - 1], "label": current_conv[sentence_id],
                                   "history_conv": sum(history_conv, []),
                                   "history_conv_utter_len": [len(history_i) for history_i in history_conv],
                                   }, new_episode

                    new_episode = False
            else:
                whole_hist_conv_scores = instance["whole_hist_conv_scores"]
                separate_hist_conv_scores = instance["separate_hist_conv_scores"]
                top_whole_hist_conv_query = instance["top_whole_hist_conv_query"]
                top_separate_hist_conv_query = instance["top_separate_hist_conv_query"]
                hist_whole_top_scores = instance["top_whole_hist_conv_scores"]
                hist_utter_top_scores = instance["top_separate_hist_conv_scores"]
                # history_sen = []
                new_episode = True
                for sentence_id in range(1, len(current_conv), 2):
                    # if not history_conv:
                    #     continue
                    # history_sen.append(current_conv[sentence_id-1])
                    # yield {"text": self.hist_utter_separator.join(history_sen), "label": current_conv[sentence_id],
                    #        "history_conv": self.hist_utter_separator.join(history_conv)}, new_episode
                    if self.msc_passage_type == "whole":
                        if not self.use_pre_calculated_score:
                            history_conv_scores = whole_hist_conv_scores[sentence_id-1] if whole_hist_conv_scores else []
                            history_conv_query = []
                        else:
                            history_conv_scores = hist_whole_top_scores[sentence_id-1][:self.n_docs] \
                                if hist_whole_top_scores else [1.0] * self.n_docs
                            history_conv_query = top_whole_hist_conv_query[sentence_id-1][:self.n_docs] \
                                if top_whole_hist_conv_query else [current_conv[sentence_id-1]] * self.n_docs
                    else:
                        if not self.use_pre_calculated_score:
                            history_conv_scores = \
                                separate_hist_conv_scores[sentence_id-1] if separate_hist_conv_scores else []
                            history_conv_query = []
                        else:
                            history_conv_scores = hist_utter_top_scores[sentence_id-1][:self.n_docs]\
                                if hist_whole_top_scores else [1.0] * self.n_docs
                            history_conv_query = top_separate_hist_conv_query[sentence_id-1][:self.n_docs] \
                                if top_separate_hist_conv_query else [current_conv[sentence_id-1]] * self.n_docs
                    if self.concat_hist_conv:
                        yield {"text": " ".join([" ".join(history_i) for history_i in history_conv]) +
                                       current_conv[sentence_id - 1] ,
                               "label": current_conv[sentence_id],
                               "history_conv": [self.hist_utter_separator.join(history_i)
                                                for history_i in history_conv],
                               "history_conv_scores": history_conv_scores,
                               "history_conv_query": history_conv_query}, new_episode
                    else:
                        yield {"text": current_conv[sentence_id - 1], "label": current_conv[sentence_id],
                               "history_conv": [self.hist_utter_separator.join(history_i)
                                                for history_i in history_conv],
                               "history_conv_scores": history_conv_scores,
                               "history_conv_query": history_conv_query}, new_episode
                    new_episode = False


class FacebookMSCTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
            cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ):
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('Twitter Conv Teacher Args')
        parser.add_argument(
            '--include_session',
            type=int,
            default=5,
            choices=[1, 2, 3, 4, 5],
            help="The included data. "
                 "1: only includes session 1 convs;"
                 "2: includes session 1 and session 2 convs;"
                 "5: includes session 1 to session 5 convs.",
        )
        parser.add_argument(
            '--persona_type',
            type=str,
            default="both_original",
            choices=["both_original", "both_revised", "none_original", "none_revised", "self_original", "self_revised",
                     "other_original", "other_revised"],
            help="The persona type of session one persona, which may not be used in the final model"
        )
        parser.add_argument(
            '--session_only',
            default=None,
            choices=[1, 2, 3, 4, 5, None],
            help="The included data. "
                 "1: only includes session 1 convs;"
                 "2: only includes session 2 convs;"
                 "5: only includes session 5 convs.",
        )
        parser.add_argument(
            '--add_persona',
            default=False,
            type=bool,
            help="Whether to append persona before the context",
        )
        parser.add_argument(
            '--concat_hist_conv',
            '-ctc',
            type=bool,
            default=False,
            help="whether to concat history conversation with context",
        )
        parser.add_argument(
            "--excluding_session1",
            type=bool,
            default=False,
            help="Whether to exclude session1 data"
        )
        FacebookMSCBaseTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        self.concat_hist_conv = opt["concat_hist_conv"]
        excluding_session1 = opt["excluding_session1"]
        if excluding_session1:
            start_idx = 1
        else:
            start_idx = 0
        if self.concat_hist_conv:
            all_tasks = [
                "facebook_msc:FacebookMSCBaseTeacher:session_id=1:concat_hist_conv=True",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=2:concat_hist_conv=True",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=3:concat_hist_conv=True",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=4:concat_hist_conv=True",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=5:concat_hist_conv=True",
            ]
        else:
            all_tasks = [
                "facebook_msc:FacebookMSCBaseTeacher:session_id=1",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=2",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=3",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=4",
                "facebook_msc:FacebookMSCBaseTeacher:session_id=5",
            ]
        include_session = opt["include_session"]
        session_only = opt["session_only"]
        opt = copy.deepcopy(opt)
        # facebook msc data does not contain the training set of session 5 conversation.
        if opt['datatype'].split(':')[0] == "train" and include_session == 5:
            include_session = 4
        if not session_only:
            opt['task'] = ','.join(all_tasks[start_idx:include_session])
        else:
            opt['task'] = all_tasks[session_only - 1]
        super().__init__(opt, shared)


class DefaultTeacher(FacebookMSCTeacher):
    pass
