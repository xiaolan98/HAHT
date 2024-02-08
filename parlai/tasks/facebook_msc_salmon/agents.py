from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from .build import build
from parlai.utils.strings import normalize_reply

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


class FacebookMSCSalmonBaseTeacher(DialogTeacher):
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
            '--few_shot',
            type=bool,
            default=False,
            help="whether to conduct few shot setting",
        )
        parser.add_argument(
            '--add_sentence_tok',
            type=bool,
            default=False,
            help="whether to add sentence token as the first token",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.session_id = opt["session_id"]
        self.id = "facebook_msc" + str(self.session_id)
        self.concat_hist_conv = opt["concat_hist_conv"]
        opt['datafile'] = _path(opt)
        self.hist_utter_separator = opt.get("hist_utter_separator", "\n")
        self.few_shot = opt.get("few_shot", False)
        self.add_sentence_tok = opt.get("add_sentence_tok", False)
        self.session_opening_only = opt.get("session_opening_only", False)
        super().__init__(opt, shared)

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

            new_history_conv = []
            for h_c in history_conv:
                h_c_temp = []
                for idx in range(1, len(h_c), 2):
                    h_c_temp.append("User: " + h_c[idx-1] + "\n" + "Assistant: " + h_c[idx])
                if len(h_c) % 2 == 1:
                    h_c_temp.append("User: " + h_c[-1])
                new_history_conv.append(h_c_temp)
            current_conv = [normalize_reply(i) for i in current_conv if i != DUMMY_TEXT]
            episode = []
            for sentence_id in range(1, len(current_conv), 2):
                if self.concat_hist_conv and sentence_id == 1:
                    episode.append(
                        {"text": self.hist_utter_separator.join(
                            [self.hist_utter_separator.join(history_i) for history_i in history_conv]
                        ) + "\n" + current_conv[sentence_id - 1],
                         "label": current_conv[sentence_id],
                         "history_conv": []
                         }
                    )
                elif self.concat_hist_conv and sentence_id != 1:
                    episode.append(
                        {"text": current_conv[sentence_id - 1],
                         "label": current_conv[sentence_id],
                         "history_conv": []
                         }
                    )
                elif not self.concat_hist_conv:
                    # if new_history_conv:
                    #     hist_conv_0 = [_i[0] for _i in sum(new_history_conv, [])]
                    #     hist_conv_1 = [_i[1] for _i in sum(new_history_conv, [])]
                    #     output_hist_conv = [hist_conv_0, hist_conv_1]
                    # else:
                    #     output_hist_conv = []
                    episode.append(
                        {"text": "User: " + current_conv[sentence_id - 1],
                         "label": "Assistant: " + current_conv[sentence_id],
                         "history_conv": sum(new_history_conv, []),
                         "history_conv_len": [len(history_i) for history_i in new_history_conv],
                         }
                    )
                if self.session_opening_only:
                    break
            data.append(episode)
        print(1)
        if self.few_shot:
            data = data[:100]
        for episode in data:
            start_idx = 0
            for i, turn in enumerate(episode):
                yield turn, i == start_idx


class FacebookMSCSalmonTeacher(MultiTaskTeacher):
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
            "--session_opening_only",
            type=bool,
            default=False,
            help="Whether to only remain session opening data"
        )
        FacebookMSCSalmonBaseTeacher.add_cmdline_args(parser, partial_opt)
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
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=1:concat_hist_conv=True",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=2:concat_hist_conv=True",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=3:concat_hist_conv=True",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=4:concat_hist_conv=True",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=5:concat_hist_conv=True",
            ]
        else:
            all_tasks = [
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=1",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=2",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=3",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=4",
                "facebook_msc_salmon:FacebookMSCSalmonBaseTeacher:session_id=5",
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


class DefaultTeacher(FacebookMSCSalmonTeacher):
    pass
