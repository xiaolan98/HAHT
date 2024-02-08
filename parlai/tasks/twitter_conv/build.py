#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

# try:
#     from emoji.unicode_codes import UNICODE_EMOJI
#     import unidecode
# except ImportError:
#     raise ImportError('Please `pip install emoji unidecode` for the twitter task.')

import io
import gzip
import parlai.core.build_data as build_data
import os
import re
import json
import copy
import pickle as pkl
from collections import defaultdict
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager


## Clean tweets
def compound_word_split(compound_word):
    """
    Split a given compound word(string) and return list of words in given compound_word
    Ex: compound_word='pyTWEETCleaner' --> ['py', 'TWEET', 'Cleaner']
    """
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', compound_word)
    return [m.group(0) for m in matches]


def remove_non_ascii_chars(text):
    """
    return text after removing non-ascii characters i.e. characters with ascii value >= 128
    """
    return ''.join([w if ord(w) < 128 else ' ' for w in text])


def remove_hyperlinks(text):
    """
    return text after removing hyperlinks
    """
    return ' '.join([w for w in text.split(' ') if not 'http' in w])


def get_cleaned_text(text):
    """
    return cleaned text(string) for provided tweet text(string)
    """
    cleaned_text = text.replace('\"', '').replace('\'', '').replace('-', ' ')

    cleaned_text = remove_non_ascii_chars(cleaned_text)

    cleaned_text = remove_hyperlinks(cleaned_text)

    cleaned_text = " ".join([token.strip() for token in cleaned_text.split() if token.strip() != ""])

    return cleaned_text


def reformat_conversation(conversations, users):
    reformatted_conv = []
    for conv, user_seq in zip(conversations, users):
        conv_temp = []
        previous_speaker = -1
        for sen, user_id in zip(conv, user_seq):
            sen_temp = ""
            for word in sen:
                if word != "<eos>" and word != "<pad>":
                    sen_temp += word + " "
            if sen_temp.strip() != '':
                if user_id != previous_speaker:
                    conv_temp.append(sen_temp.strip())
                    previous_speaker = user_id
                else:
                    conv_temp[-1] += " " + sen_temp.strip()
        reformatted_conv.append(conv_temp)
    return reformatted_conv


def clean_conv(conversations):
    cleaned_conversations = []
    for conv in conversations:
        cleaned_conv = []
        for utter in conv:
            cleaned_utter = get_cleaned_text(utter)
            cleaned_conv.append(cleaned_utter)
        cleaned_conversations.append(cleaned_conv)
    return cleaned_conversations


def save_data(conversations, users, data_type, max_session_num=5):
    session_conv_num = [0] * max_session_num
    session_conv_idx = [[] for _ in range(max_session_num)]
    user_pair_num_dict = defaultdict(int)
    user_pair_idx_dict = defaultdict(list)
    for idx, user in enumerate(users):
        user_pair_num_dict[tuple(set(user))] += 1
        user_pair_idx_dict[tuple(set(user))].append(idx)
        assert len(tuple(set(user))) == 2

    reformatted_convs = reformat_conversation(conversations, users)
    reformatted_convs = clean_conv(reformatted_convs)
    for user_pair in user_pair_num_dict:
        pair_num = user_pair_num_dict[user_pair]
        pair_convs_idx = user_pair_idx_dict[user_pair]
        if pair_num <= max_session_num:
            for i in range(pair_num):
                session_conv_num[i] += 1
                session_conv_idx[i].append(pair_convs_idx[:i + 1])
        else:
            # session_conv_num[max_session_num-1] += int(pair_num / max_session_num)
            for i in range(max_session_num):
                session_conv_num[i] += int(pair_num / max_session_num)
                for j in range(int(pair_num / max_session_num)):
                    session_conv_idx[i].append(pair_convs_idx[max_session_num * j:max_session_num * j + i + 1])

            rest_pair_num = int(pair_num % max_session_num)
            if rest_pair_num != 0:
                # session_conv_num[rest_pair_num-1] += 1
                for i in range(rest_pair_num):
                    session_conv_num[i] += 1
                    session_conv_idx[i].append(pair_convs_idx[-i - 1:])
    session_convs = [[] for _ in range(max_session_num)]
    for _i in range(max_session_num):
        for conversation_idx in session_conv_idx[_i]:
            user_pair = list(set(users[conversation_idx[0]]))
            initiator_id = users[conversation_idx[-1]][0]
            history_conv = [reformatted_convs[_j] for _j in conversation_idx[:-1]]
            current_conv = reformatted_convs[conversation_idx[-1]]
            conv_json = {"user_ids": user_pair, "initiator_id": initiator_id,
                         "history_conv": history_conv, "current_conv": current_conv}
            session_convs[_i].append(conv_json)
    return session_convs


def save_json_list(json_list, data_path):
    with open(data_path, "w") as f:
        for json_item in json_list:
            f.write(json.dumps(json_item) + "\n")


def reformat_data(root_data_path, root_save_path, max_session_num=5):
    if not os.path.exists(root_save_path):
        os.mkdir(root_save_path)
    # training data
    training_conversations = pkl.load(open(os.path.join(root_data_path, "train", "convs.pkl"), "rb"))
    training_users = pkl.load(open(os.path.join(root_data_path, "train", "convs_users.pkl"), "rb"))
    train_session_convs = save_data(training_conversations, training_users, "train")
    # valid data
    valid_conversations = pkl.load(open(os.path.join(root_data_path, "valid", "convs.pkl"), "rb"))
    valid_users = pkl.load(open(os.path.join(root_data_path, "valid", "convs_users.pkl"), "rb"))
    valid_session_convs = save_data(valid_conversations, valid_users, "valid")

    # test data
    testing_conversations = pkl.load(open(os.path.join(root_data_path, "test", "convs.pkl"), "rb"))
    testing_users = pkl.load(open(os.path.join(root_data_path, "test", "convs_users.pkl"), "rb"))
    test_session_convs = save_data(testing_conversations, testing_users, "test")

    all_session_convs = [[] for _ in range(max_session_num)]
    for _i in range(max_session_num):
        all_session_convs[_i] = train_session_convs[_i] + valid_session_convs[_i] + test_session_convs[_i]
    for _i in range(max_session_num):
        session_data_path = os.path.join(root_save_path, "session"+str(_i+1))
        if not os.path.exists(session_data_path):
            os.mkdir(session_data_path)
        valid_set_len = int(len(all_session_convs[_i]) * 0.1)
        train_set_idx = len(all_session_convs[_i]) - valid_set_len * 2
        save_json_list(all_session_convs[_i][:train_set_idx], os.path.join(session_data_path, "train_convs.txt"))
        save_json_list(all_session_convs[_i][train_set_idx: train_set_idx + valid_set_len],
                       os.path.join(session_data_path, "valid_convs.txt"))
        save_json_list(all_session_convs[_i][-valid_set_len:], os.path.join(session_data_path, "test_convs.txt"))

    #     session_data_path = os.path.join(save_path_root, "session"+str(_i+1))
    #     if not os.path.exists(session_data_path):
    #         os.mkdir(session_data_path)
    #     with open(os.path.join(session_data_path, data_type+"_convs.txt"), "w") as f:


def build(opt):
    dpath = os.path.join(opt['datapath'], 'twitter_conv')
    version = '2.0'
    original_data_path = os.path.join(opt['datapath'], 'TwitterCon', 'tc_small')
    cleaned_data_path = os.path.join(opt['datapath'], 'TwitterCon', 'tc_small_cleaned')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        reformat_data(original_data_path, os.path.join(dpath, "original"), opt["max_session"])
        reformat_data(cleaned_data_path, os.path.join(dpath, "cleaned"), opt["max_session"])

        # Mark the data as built.
        build_data.mark_done(dpath, version)

#
# if __name__ == '__main__':
#     path_a = "../../../data/TwitterCon/tc_small_cleaned/"
#     path_b = "./temp"
#     reformat_data(path_a, path_b)