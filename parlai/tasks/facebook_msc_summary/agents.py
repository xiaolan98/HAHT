from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from .build import build
from parlai.utils.strings import uppercase, normalize_reply
import nltk

import json
import copy
import os
from random import random
import string

DUMMY_TEXT = '__SILENCE__'


def get_verb_dicts(path):
    dict1 = {}
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            key = line[0]
            value = line[1]
            dict1[key] = value
    return dict1


# def normalize_reply(text: str, version=1) -> str:
#     """
#     Standardize the capitalization and punctuation spacing of the input text.
#
#     Version 1: Fix sentence start casing, and punctuation.
#
#     Version 2: Add trailing period, if missing.
#     """
#
#     switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]
#
#     # add spaces so that words and punctuation can be seaprated
#     # new_text = text.lower()
#     new_text = text
#
#     # normalize in case of human:
#     for new, old in switch_list:
#         new_text = new_text.replace(old, new).replace('  ', ' ')
#
#     # split on punctuation to find sentence boundaries
#     # capitalize stuff
#     tokens = new_text.split(' ')
#     for i in range(len(tokens)):
#         if i == 0:
#             tokens[i] = uppercase(tokens[i])
#         elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
#             tokens[i] = uppercase(tokens[i])
#         elif tokens[i] in '?.!' and i < len(tokens) - 1:
#             tokens[i + 1] = uppercase(tokens[i + 1])
#     new_text = ' '.join(tokens)
#     new_text = ' ' + new_text + ' '
#
#     for tup in switch_list:
#         new_text = new_text.replace(tup[0], tup[1])
#
#     # get rid of surrounding whitespace
#     new_text = new_text.strip()
#     new_text = new_text.replace('  ', ' ')
#
#     if version > 1 and new_text and new_text[-1] not in '!.?)"\'':
#         new_text += '.'
#
#     return new_text


def convert_from_first_person2second_person(text, verb_map, remains_map):
    switch_dict = {
        'my': 'his',
        'me': 'him',
        'mine': 'his',
        'myself': 'himself',
        'we': "they"
    }
    token_list = nltk.word_tokenize(text)
    tok_tag = nltk.pos_tag(token_list)
    new_token_list = []
    counter = 0
    for idx, token in enumerate(token_list):
        if token.lower() in switch_dict:
            new_token_list.append(switch_dict[token.lower()])
        elif token.lower() == 'i':
            new_token_list.append("The user")
            if tok_tag[idx + 1][1].startswith("RB"):
                if tok_tag[idx + 2][1].startswith("RB"):
                    if tok_tag[idx + 3][1].startswith("VB") \
                            and tok_tag[idx + 3][1] != "VBD" and tok_tag[idx + 3][1] != "VBN":
                        if tok_tag[idx + 3][1] != "VBZ" and tok_tag[idx + 3][1] != "VBG":
                            verb_phrase = (token_list[idx + 1]+" "+token_list[idx + 2]+" "+token_list[idx + 3]).lower()
                            new_token_list.append(verb_map[verb_phrase])
                            counter = 3
                    elif not tok_tag[idx + 3][1].startswith("VB"):
                        verb_phrase = (
                                    token_list[idx + 1] + " " + token_list[idx + 2] + " " + token_list[idx + 3]).lower()
                        new_token_list.append(remains_map[verb_phrase])
                        counter = 3
                else:
                    if tok_tag[idx + 2][1].startswith("VB") \
                            and tok_tag[idx + 2][1] != "VBD" and tok_tag[idx + 2][1] != "VBN":
                        if not tok_tag[idx + 2][1] == "VBZ" and not tok_tag[idx + 2][1] == "VBG":
                            verb_phrase = (token_list[idx + 1] + " " + token_list[idx + 2]).lower()
                            new_token_list.append(verb_map[verb_phrase])
                            counter = 2
                    elif not tok_tag[idx + 2][1].startswith("VB"):
                        verb_phrase = (token_list[idx + 1] + " " + token_list[idx + 2]).lower()
                        new_token_list.append(remains_map[verb_phrase])
                        counter = 2
            else:
                if tok_tag[idx + 1][1].startswith("VB") \
                        and tok_tag[idx + 1][1] != "VBD" and tok_tag[idx + 1][1] != "VBN":
                    if not tok_tag[idx + 1][1] == "VBZ" and not tok_tag[idx + 1][1] == "VBG":
                        new_token_list.append(verb_map[token_list[idx + 1].lower()])
                        counter = 1
                elif not tok_tag[idx + 1][1].startswith("VB"):
                    try:
                        new_token_list.append(remains_map[token_list[idx + 1].lower()])
                    except KeyError:
                        new_token_list.append(token_list[idx + 1])
                        print(token_list[idx+1])
                    counter = 1
        else:
            if counter <= 0:
                new_token_list.append(token)
            else:
                counter -= 1
    new_text = " ".join(new_token_list)
    return new_text


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    session_name = "session_" + str(opt["session_id"])
    # return os.path.join(opt['datapath'], 'facebook_msc', session_name, dt + '_convs_w_scores.txt')
    return os.path.join(opt['datapath'], 'facebook_msc_original/msc/msc_personasummary', session_name, dt + '.txt')


class FacebookMSCSummaryBaseTeacher(DialogTeacher):
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
            '--prepend_persona',
            type=bool,
            default=False,
            help="whether to prepend the persona into the conversation utterance",
        )
        parser.add_argument(
            '--few_shot',
            type=bool,
            default=False,
            help="whether to prepend the persona into the conversation utterance",
        )
        parser.add_argument(
            '--consider_no_summary',
            type=bool,
            default=False,
            help="whether to consider the case that there's no fact for the conversation utterances",
        )
        parser.add_argument(
            '--change_person_token',
            type=bool,
            default=False,
            help="whether to change the person token",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.session_id = opt["session_id"]
        self.id = "facebook_msc" + str(self.session_id)
        self.prepend_persona = opt.get("prepend_persona", False)
        self.change_person_token = opt.get("change_person_token", False)
        self.few_shot = opt.get("few_shot", False)
        self.consider_no_summary = opt.get("consider_no_summary", False)
        self.opt = opt
        opt['datafile'] = _path(opt)
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        instances = []
        with open(path) as json_file:
            for line in json_file.readlines():
                instances.append(json.loads(line))
        if self.change_person_token:
            verb_map = get_verb_dicts("../verb_dict.txt")
            remains_map = get_verb_dicts("../remains.txt")
        data = []
        for instance in instances:
            episode = []
            dialogue = instance["dialog"]
            for item in dialogue:
                conv_utter = item["text"]
                bot_id = item['id']
                summary = item['persona_text'].strip() if 'persona_text' in item else ""
                if summary == "":
                    if self.consider_no_summary:
                        if random() < 0.3:
                            summary = "NO_SUMMARY"
                        else:
                            continue
                    else:
                        continue
                summary = normalize_reply(summary)
                conv_utter = normalize_reply(conv_utter) + "\n"
                # conv_utter = conv_utter + "\n"
                if self.change_person_token:
                    summary_second = convert_from_first_person2second_person(summary, verb_map, remains_map)
                else:
                    summary_second = ""
                episode.append(
                    {
                        "text": conv_utter,
                        "label": summary,
                        "label_second": summary_second,
                        "bot_id": bot_id
                    }
                )
            data.append(episode)
        output_data = []
        for episode in data:
            for idx, utter_dict in enumerate(episode):
                text = utter_dict["text"]
                label = utter_dict["label"]
                label_second = utter_dict["label_second"]
                if idx < len(episode) - 1:
                    next_utter_dict = episode[idx + 1]
                    next_text = next_utter_dict["text"]
                    next_label = next_utter_dict["label"]
                    next_label_second = next_utter_dict["label_second"]
                    if self.change_person_token:
                        output_data.append(
                            {
                                "text": "User: " + text + "Assistant: " + next_text,
                                "label": label_second + " " + next_label
                                # "label": label + " " + next_label_second
                            }
                        )
                    else:
                        output_data.append(
                            {
                                "text": "User: " + text + "Assistant: " + next_text,
                                "label": label + " " + next_label
                            }
                        )
                output_data.append(
                    {
                        "text": "Assistant: " + text,
                        "label": label
                    }
                )
                if self.change_person_token:
                    output_data.append(
                        {
                            "text": "User: " + text,
                            "label": label_second
                        }
                    )
        data = output_data
        print(1)
        if self.few_shot:
            dt = self.opt['datatype'].split(':')[0]
            if dt == "train":
                print("Only use 200 conversation to do the few shot training (%d)" % 200)
                data = data[:200]
        for turn in data:
            yield turn, True


class FacebookMSCSummaryTeacher(MultiTaskTeacher):
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
            '--add_segmentation',
            default=False,
            type=bool,
            help="whether to separate the user 1 data with user 2 data",
        )
        FacebookMSCSummaryBaseTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        all_tasks = [
            "facebook_msc_summary:FacebookMSCSummaryBaseTeacher:session_id=1",
            "facebook_msc_summary:FacebookMSCSummaryBaseTeacher:session_id=2",
            "facebook_msc_summary:FacebookMSCSummaryBaseTeacher:session_id=3",
            "facebook_msc_summary:FacebookMSCSummaryBaseTeacher:session_id=4",
        ]
        include_session = opt["include_session"]
        session_only = opt["session_only"]
        opt = copy.deepcopy(opt)
        # facebook msc data does not contain the training set of session 5 conversation.
        if opt['datatype'].split(':')[0] == "train" and include_session == 4:
            include_session = 3
        if not session_only:
            opt['task'] = ','.join(all_tasks[:include_session])
        else:
            opt['task'] = all_tasks[session_only - 1]
        super().__init__(opt, shared)


class DefaultTeacher(FacebookMSCSummaryTeacher):
    pass
