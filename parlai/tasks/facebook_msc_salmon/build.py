#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
import json
from parlai.utils.logging import logger
from parlai.core.build_data import DownloadableFile


MSC_DATASETS_VERSION = 'v0.1'


RESOURCES = [
    DownloadableFile(
        f'http://parl.ai/downloads/msc/msc_{MSC_DATASETS_VERSION}.tar.gz',
        f'msc_{MSC_DATASETS_VERSION}.tar.gz',
        'e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60',
    ),
    DownloadableFile(
        'http://parl.ai/downloads/convai2/convai2_fix_723.tgz',
        'convai2_fix_723.tgz',
        'd0ae89defe2fd0b0a4221eaa642a457d7d40cef475f54798119c7f3b8dd9361d',
    )
]


def check_original_data(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(data_path)
    else:
        logger.warning("Original facebook msc data is downloaded.")


def get_msc_dir_path(opt):
    dpath = os.path.join(opt['datapath'], 'facebook_msc')
    return dpath


def reformat_session1_conv(data_path, save_path):
    with open(data_path, 'r') as f, open(save_path, "w") as fw:
        conversation = []
        your_persona = []
        partner_persona = []
        for line in f:
            if line.startswith("1 "):
                if conversation != []:
                    conv_json = {
                        "history_conv": [],
                        "current_conv": conversation,
                        "your_persona": your_persona,
                        "partner_persona": partner_persona
                    }
                    fw.write(json.dumps(conv_json) + "\n")
                conversation = []
                your_persona = []
                partner_persona = []
            line = " ".join(line.split(" ")[1:])
            if line.startswith("your persona"):
                your_persona.append(line[14:])
            elif line.startswith("partner's persona"):
                partner_persona.append(line[19:])
            else:
                line = line.split("\t")
                conversation.append(line[0])
                conversation.append(line[1])


def reformat_session1_data(session1_root_path, root_save_path, persona_type="both_original"):
    # PersonaChat only contains train and valid data. I set test is the same as valid
    # persona type can be "both_original" or "both_revised"
    if not os.path.exists(root_save_path):
        os.mkdir(root_save_path)
    # training set
    train_data_path = os.path.join(session1_root_path, "train_"+persona_type+"_no_cands.txt")
    train_save_path = os.path.join(root_save_path, "train_convs.txt")
    reformat_session1_conv(train_data_path, train_save_path)
    # valid set
    valid_data_path = os.path.join(session1_root_path, "valid_"+persona_type+"_no_cands.txt")
    valid_save_path = os.path.join(root_save_path, "valid_convs.txt")
    reformat_session1_conv(valid_data_path, valid_save_path)
    # testing set
    test_data_path = os.path.join(session1_root_path, "valid_"+persona_type+"_no_cands.txt")
    test_save_path = os.path.join(root_save_path, "test_convs.txt")
    reformat_session1_conv(test_data_path, test_save_path)


def conv_json2list(dialog):
    previous_speaker = ""
    dialog_out = []
    for d in dialog:
        current_speaker = d["id"]
        if current_speaker == previous_speaker and previous_speaker != "":
            dialog_out[-1] += d['text']
            logger.warning("Facebook MSC conversation needs to be used carefully.")
        else:
            dialog_out.append(d["text"])
        previous_speaker = current_speaker
    return dialog_out


def reformat_session2345_conv(data_path, save_path):
    with open(data_path, "r") as f, open(save_path, "w") as fw:
        for line in f:
            conv_json = {}
            line = json.loads(line)
            conv_json["your_persona"] = line["personas"][0]
            conv_json["partner_persona"] = line["personas"][1]
            conv_json["current_conv"] = conv_json2list(line['dialog'])
            history_conv = []
            for _pd in line["previous_dialogs"]:
                hist_conv_tmp = []
                for d_line in _pd["dialog"]:
                    hist_conv_tmp.append(d_line["text"].strip())
                history_conv.append(hist_conv_tmp)
            conv_json["history_conv"] = history_conv
            fw.write(json.dumps(conv_json) + "\n")


def reformat_session2345_data(session_root_path, save_root_path, session_id=2):
    # facebook multi-session dataset
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    # training set
    if session_id != 5:
        train_data_path = os.path.join(session_root_path, "train.txt")
        train_save_path = os.path.join(save_root_path, "train_convs.txt")
        reformat_session2345_conv(train_data_path, train_save_path)
    # valid set
    valid_data_path = os.path.join(session_root_path, "valid.txt")
    valid_save_path = os.path.join(save_root_path, "valid_convs.txt")
    reformat_session2345_conv(valid_data_path, valid_save_path)
    # testing set
    test_data_path = os.path.join(session_root_path, "test.txt")
    test_save_path = os.path.join(save_root_path, "test_convs.txt")
    reformat_session2345_conv(test_data_path, test_save_path)


def reformat_data(root_data_path, root_save_path, persona_type="both_original"):
    if not os.path.exists(root_save_path):
        os.mkdir(root_save_path)
    # session 1 data
    session1_root_save_path = os.path.join(root_save_path, "session1")
    reformat_session1_data(root_data_path, session1_root_save_path, persona_type)
    # session 2 data
    session2_root_path = os.path.join(root_data_path, "msc", "msc_dialogue", "session_2")
    session2_root_save_path = os.path.join(root_save_path, "session2")
    reformat_session2345_data(session2_root_path, session2_root_save_path, 2)
    # session 3 data
    session3_root_path = os.path.join(root_data_path, "msc", "msc_dialogue", "session_3")
    session3_root_save_path = os.path.join(root_save_path, "session3")
    reformat_session2345_data(session3_root_path, session3_root_save_path, 3)
    # session 4 data
    session4_root_path = os.path.join(root_data_path, "msc", "msc_dialogue", "session_4")
    session4_root_save_path = os.path.join(root_save_path, "session4")
    reformat_session2345_data(session4_root_path, session4_root_save_path, 4)
    # session 5 data
    session5_root_path = os.path.join(root_data_path, "msc", "msc_dialogue", "session_5")
    session5_root_save_path = os.path.join(root_save_path, "session5")
    reformat_session2345_data(session5_root_path, session5_root_save_path, 5)


def build(opt):
    version = MSC_DATASETS_VERSION
    # create particular instance of dataset depending on flags..
    dpath = get_msc_dir_path(opt)
    original_data_path = os.path.join(opt["datapath"], "facebook_msc_original")
    if not build_data.built(dpath, version):
        logger.warning('[build data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        # check whether the original data exists
        check_original_data(original_data_path)
        # build new facebook msc data
        reformat_data(original_data_path, dpath, opt["persona_type"])
        # Mark the data as built.
        build_data.mark_done(dpath, version)

    return dpath


# if __name__ == '__main__':
#     # session 1
#     _root_save_path = os.path.join("/home/zhangtong/ParlAI/data/", 'facebook_msc')
#     _session1_root_save_path = os.path.join(_root_save_path, "session1")
#     _original_data_path = os.path.join("/home/zhangtong/ParlAI/data/", "facebook_msc_original")
#     reformat_session1_data(_original_data_path, _session1_root_save_path, "both_original")
