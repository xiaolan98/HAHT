from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from .build import build
from parlai.utils.strings import normalize_reply
from parlai.utils.data import DatatypeHelper

import json
import copy
import os
import string


DUMMY_TEXT = '__SILENCE__'


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    session_name = "session" + str(opt["session_id"])
    # return os.path.join(opt['datapath'], 'facebook_msc', session_name, dt + '_convs_w_scores.txt')
    return os.path.join(opt['datapath'], 'facebook_msc', session_name, dt + '_convs.txt')


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
        parser.add_argument(
            '--add_user_token',
            '-aut',
            type=bool,
            default=False,
            help="whether to prepend user or assistant tokens to the conversation tokens",
        )
        parser.add_argument(
            '--add_persona',
            type=bool,
            default=False,
            help="whether to add your persona: and partner's persona: ",
        )
        parser.add_argument(
            '--history_summary_type',
            type=str,
            default="self",
            choices=["self", "others", "both"],
            help="the persona chosen to use",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.session_id = opt["session_id"]
        self.id = "facebook_msc" + str(self.session_id)
        self.concat_hist_conv = opt["concat_hist_conv"]
        opt['datafile'] = _path(opt)
        self.msc_passage_type = opt.get("msc_passage_type", "whole")
        self.hist_utter_separator = opt.get("hist_utter_separator", "\n")
        self.add_user_token = opt.get("add_user_token", False)
        if self.add_user_token and not shared:
            print("Add special tokens ['User: ', 'Assistant: '] to each sentence")
        self.use_persona_as_history = opt.get("use_persona_as_history", False)
        self.session_opening_only = opt.get("session_opening_only", False)
        self.add_persona = opt.get("add_persona", False)
        self.history_summary_type = opt.get("history_summary_type", "self")
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
            # user_pair = instance["user_ids"]
            history_conv = instance["history_conv"]
            current_conv = instance["current_conv"]
            your_persona = instance["your_persona"]
            partner_persona = instance["partner_persona"]
            if self.add_persona:
                for idx, persona in enumerate(your_persona):
                    your_persona[idx] = "your persona: " + persona
                for idx, persona in enumerate(partner_persona):
                    partner_persona[idx] = "partner's persona: " + persona

            if history_conv:
                new_history_conv = []
                for hist_c in history_conv:
                    temp_hist_c = []
                    for user_id, hist_utter in enumerate(hist_c[:10]):
                        if hist_utter != DUMMY_TEXT:
                            new_hist_utter = normalize_reply(hist_utter)
                        else:
                            new_hist_utter = hist_utter
                        if self.add_user_token:
                            new_hist_utter = "User: " + new_hist_utter if user_id % 2 == 0 else \
                                "Assistant: " + new_hist_utter
                        temp_hist_c.append(new_hist_utter)
                    new_history_conv.append(temp_hist_c)
                history_conv = new_history_conv
                # history_conv = [[normalize_reply(j) for j in i if j != DUMMY_TEXT] for i in history_conv]
            if self.use_persona_as_history:
                if self.history_summary_type == "both":
                    history_conv = [your_persona, partner_persona]
                elif self.history_summary_type == "self":
                    history_conv = [your_persona]
                elif self.history_summary_type == "others":
                    history_conv = [partner_persona]

                new_history_conv = []
                for user_id, hist_c in enumerate(history_conv):
                    temp_hist_c = []
                    for hist_utter in hist_c:
                        if hist_utter != DUMMY_TEXT:
                            new_hist_utter = normalize_reply(hist_utter)
                        else:
                            new_hist_utter = hist_utter
                        if self.add_user_token:
                            new_hist_utter = "User: " + new_hist_utter if user_id % 2 == 0 else \
                                "Assistant: " + new_hist_utter
                        temp_hist_c.append(new_hist_utter)
                    new_history_conv.append(temp_hist_c)
                history_conv = new_history_conv
            new_current_conv = []
            for user_id, current_conv_utter in enumerate(current_conv):
                if current_conv_utter != DUMMY_TEXT:
                    new_current_conv_utter = normalize_reply(current_conv_utter)
                else:
                    new_current_conv_utter = current_conv_utter
                if self.add_user_token:
                    new_current_conv_utter = "User: " + new_current_conv_utter if user_id % 2 == 0 else\
                        "Assistant: " + new_current_conv_utter
                new_current_conv.append(new_current_conv_utter)
            current_conv = new_current_conv
            # current_conv = [normalize_reply(i) for i in current_conv if i != DUMMY_TEXT]
            episode = []
            for sentence_id in range(1, len(current_conv), 2):
                if self.concat_hist_conv and sentence_id == 1:
                    if not history_conv:
                        episode.append(
                            {"text": current_conv[sentence_id - 1],
                             "label": current_conv[sentence_id],
                             "history_conv": []
                             }
                        )
                    else:
                        episode.append(
                            {"text": self.hist_utter_separator.join(
                                [self.hist_utter_separator.join(history_i) for history_i in history_conv]
                            ) + "\n" + current_conv[sentence_id - 1],
                             "label": current_conv[sentence_id],
                             "history_conv": []
                             }
                        )
                else:
                    if self.msc_passage_type == "whole":
                        episode.append(
                            # {"text": "\n" + current_conv[sentence_id - 1],
                            {"text": current_conv[sentence_id - 1] + "\n",
                             "label": current_conv[sentence_id],
                             "history_conv": [self.hist_utter_separator.join(history_i)
                                              for history_i in history_conv],
                             }
                        )
                    else:
                        assert self.msc_passage_type == "separate"
                        episode.append(
                            {"text": current_conv[sentence_id - 1] + '\n',
                             "label": current_conv[sentence_id],
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
            "--excluding_session1",
            type=bool,
            default=False,
            help="Whether to exclude session1 data"
        )
        parser.add_argument(
            "--use_persona_as_history",
            type=bool,
            default=False,
            help="Whether to use the persona as history conversation to check the upper bound"
        )
        parser.add_argument(
            "--session_opening_only",
            type=bool,
            default=False,
            help="Whether to only remain session opening data"
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
