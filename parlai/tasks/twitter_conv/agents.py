#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.utils.strings import normalize_reply
from parlai.core.opt import Opt
from .build import build
from parlai.utils.data import DatatypeHelper

import json
import copy
import os


def _path(opt, conv_type):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    session_name = "session" + str(opt["session_id"])
    return os.path.join(opt['datapath'], 'twitter_conv', conv_type, session_name, dt+'_convs.txt')


class TwitterConvBaseTeacher(DialogTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ):
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('Twitter Conversation Teacher Args')
        parser.add_argument(
            '--conv_type',
            '-ct',
            type=str,
            default="cleaned",
            choices=["cleaned", "original"],
            help="whether to include session 1 (convai2:normalized)",
        )
        parser.add_argument(
            '--session_id',
            '-si',
            type=int,
            default="2",
            help="session id. Session id should be from 1-5",
        )
        parser.add_argument(
            '--max_session',
            '-ms',
            type=int,
            default=5,
            help="Maximum session number, default number is the same as the mas dataset",
        )
        parser.add_argument(
            '--concat_hist_conv',
            '-ctc',
            type=bool,
            default=False,
            help="whether to concat history conversation with context",
        )
        parser.add_argument(
            '--add_user_token',
            '-aut',
            type=bool,
            default=False,
            help="whether to prepend user or assistant tokens to the conversation tokens",
        )
        return parser

    def __init__(self, opt, shared=None):
        if ':' in opt['task']:
            conv_type = "original" if "original" in opt['task'] else "cleaned"
        else:
            conv_type = opt.get("conv_type", "cleaned")
        self.session_id = opt["session_id"]
        self.id = "twitter_conv" + str(self.session_id)
        self.session_id = opt["session_id"]
        self.concat_hist_conv = opt["concat_hist_conv"]
        self.hist_utter_separator = opt.get("hist_utter_separator", " ")
        self.msc_passage_type = opt.get("msc_passage_type", "separate")
        self.session_opening_only = opt.get("session_opening_only", False)
        self.add_user_token = opt.get("add_user_token", False)
        if self.add_user_token and not shared:
            print("Add special tokens ['User: ', 'Assistant: '] to each sentence")
        opt['datafile'] = _path(opt, conv_type)
        super().__init__(opt, shared)

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        self.metrics.clear_recent()
        if hasattr(self, 'lastY') and self.lastY is not None:
            if not DatatypeHelper.is_training(self.datatype) and \
                    (self.lastY[0].startswith("Assistant:") or self.lastY[0].startswith("assistant:")):
                self.lastY = (" ".join(self.lastY[0].split(" ")[1:]), )
            self.metrics.evaluate_response(observation, self.lastY)
            self.custom_evaluation(self.last_act, self.lastY, observation)
            self.lastY = None
        recent_metrics = self.metrics.report_recent()
        if recent_metrics:
            # for display purposes (display_model), take all accumulated
            # metrics back into the original observation. This is an abuse of
            # Messages being pointers
            if 'metrics' in observation:
                # override agent-level metrics if present
                observation.pop('metrics')
            observation['metrics'] = recent_metrics
        return observation

    def setup_data(self, path):
        print('loading: ' + path)
        instances = []
        with open(path) as json_file:
            for line in json_file.readlines():
                instances.append(json.loads(line))
        data = []
        for instance in instances:
            user_pair = instance["user_ids"]
            history_conv = instance["history_conv"]
            current_conv = instance["current_conv"]
            if history_conv:
                new_history_conv = []
                for hist_c in history_conv:
                    temp_hist_c = []
                    for user_id, hist_utter in enumerate(hist_c):
                        new_hist_utter = normalize_reply(hist_utter)
                        if self.add_user_token:
                            new_hist_utter = "User: " + new_hist_utter if user_id % 2 == 0 else \
                                "Assistant: " + new_hist_utter
                        temp_hist_c.append(new_hist_utter)
                    new_history_conv.append(temp_hist_c)
                history_conv = new_history_conv
            new_current_conv = []
            for user_id, current_conv_utter in enumerate(current_conv):
                new_current_conv_utter = normalize_reply(current_conv_utter)
                if self.add_user_token:
                    new_current_conv_utter = "User: " + new_current_conv_utter if user_id % 2 == 0 else \
                        "Assistant: " + new_current_conv_utter
                new_current_conv.append(new_current_conv_utter)
            current_conv = new_current_conv
            # current_conv = [normalize_reply(i) for i in current_conv]
            # new_episode = True
            episode = []
            for sentence_id in range(1, len(current_conv), 2):
                if self.concat_hist_conv and sentence_id == 1:
                    episode.append(
                        {"text": "\n".join([" ".join(history_i) for history_i in history_conv]) + "\n" +
                                 current_conv[sentence_id - 1],
                         "label": current_conv[sentence_id],
                         "history_conv": []}
                    )
                else:
                    if self.msc_passage_type == "whole":
                        episode.append(
                            {"text": current_conv[sentence_id - 1], "label": current_conv[sentence_id],
                             "history_conv": ["\n".join(history_i) for history_i in history_conv]}
                        )
                    else:
                        assert self.msc_passage_type == "separate", self.msc_passage_type
                        episode.append(
                            {"text": current_conv[sentence_id - 1], "label": current_conv[sentence_id],
                             "history_conv": [h + "\n" for h in sum(history_conv, [])],
                             "history_conv_utter_len": [len(history_i) for history_i in history_conv],
                             }
                        )
                if self.session_opening_only:
                    break
            data.append(episode)
        print(1)
        for episode in data:
            start_idx = 0
            for i, turn in enumerate(episode):
                yield turn, i == start_idx


class TwitterConvTeacher(MultiTaskTeacher):
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
            '--session_only',
            default=None,
            choices=[1, 2, 3, 4, 5, None],
            help="The included data. "
                 "1: only includes session 1 convs;"
                 "2: only includes session 2 convs;"
                 "5: only includes session 5 convs.",
        )
        parser.add_argument(
            '--concat_hist_conv',
            '-ctc',
            type=bool,
            default=False,
            help="whether to concat history conversation with context",
        )
        parser.add_argument(
            "--session_opening_only",
            type=bool,
            default=False,
            help="Whether to only remain session opening data"
        )
        parser.add_argument(
            "--excluding_session1",
            type=bool,
            default=False,
            help="Whether to exclude session1 data"
        )
        TwitterConvBaseTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        excluding_session1 = opt["excluding_session1"]
        include_session = opt["include_session"]
        session_only = opt["session_only"]
        if excluding_session1:
            start_idx = 1
        else:
            start_idx = 0
        all_tasks = [
            "twitter_conv:TwitterConvBaseTeacher:session_id=1",
            "twitter_conv:TwitterConvBaseTeacher:session_id=2",
            "twitter_conv:TwitterConvBaseTeacher:session_id=3",
            "twitter_conv:TwitterConvBaseTeacher:session_id=4",
            "twitter_conv:TwitterConvBaseTeacher:session_id=5",
        ]
        opt = copy.deepcopy(opt)
        if not session_only:
            opt['task'] = ','.join(all_tasks[start_idx: include_session])
        else:
            opt['task'] = all_tasks[session_only-1]
        super().__init__(opt, shared)


class DefaultTeacher(TwitterConvTeacher):
    pass
