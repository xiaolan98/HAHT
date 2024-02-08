# previous version of build -- only consider two session conversations

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
import json
import copy
import pickle as pkl
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager


def save_data(conversations, users, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        previous_user_pair = set()
        previous_conv = []
        for idx, conversation in enumerate(conversations):
            conv_json = {"user_ids": list(set(users[idx])), "initiator_id": users[idx][0],
                         "history_conv": [], "current_conv": []}
            if set(users[idx]) == previous_user_pair:
                conv_json["history_conv"] = previous_conv
            else:
                previous_user_pair = set(users[idx])
            previous_speaker = -1
            for sen_idx, sentence in enumerate(conversation):
                current_speaker = users[idx][sen_idx]
                sen_temp = ""
                for word in sentence:
                    if word != "<eos>" and word != "<pad>":
                        sen_temp += word + " "
                if sen_temp.strip() != '':
                    if current_speaker != previous_speaker:
                        conv_json["current_conv"].append(sen_temp.strip())
                        previous_speaker = current_speaker
                    else:
                        conv_json["current_conv"][-1] += " " + sen_temp.strip()
            # previous_conv = copy.deepcopy(conv_json["current_conv"])
            previous_conv = conv_json["current_conv"]
            f.write(json.dumps(conv_json) + "\n")


def reformat_data(root_data_path, root_save_path):
    if not os.path.exists(root_save_path):
        os.mkdir(root_save_path)
    # training data
    training_conversations = pkl.load(open(os.path.join(root_data_path, "train", "convs.pkl"), "rb"))
    training_users = pkl.load(open(os.path.join(root_data_path, "train", "convs_users.pkl"), "rb"))
    save_data(training_conversations, training_users, os.path.join(root_save_path, "train_convs.txt"))
    # valid data
    valid_conversations = pkl.load(open(os.path.join(root_data_path, "valid", "convs.pkl"), "rb"))
    valid_users = pkl.load(open(os.path.join(root_data_path, "valid", "convs_users.pkl"), "rb"))
    save_data(valid_conversations, valid_users, os.path.join(root_save_path, "valid_convs.txt"))

    # test data
    testing_conversations = pkl.load(open(os.path.join(root_data_path, "test", "convs.pkl"), "rb"))
    testing_users = pkl.load(open(os.path.join(root_data_path, "test", "convs_users.pkl"), "rb"))
    save_data(testing_conversations, testing_users, os.path.join(root_save_path, "test_convs.txt"))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'twitter_conv')
    version = '0.4'
    original_data_path = os.path.join(opt['datapath'], 'TwitterCon', 'tc_small')
    cleaned_data_path = os.path.join(opt['datapath'], 'TwitterCon', 'tc_small_cleaned')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        reformat_data(original_data_path, os.path.join(dpath, "original"))
        reformat_data(cleaned_data_path, os.path.join(dpath, "cleaned"))

        # Mark the data as built.
        build_data.mark_done(dpath, version)
