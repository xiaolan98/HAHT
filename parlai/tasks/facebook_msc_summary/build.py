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
        # logger.warning("Original facebook msc data is downloaded.")
        pass


def build(opt):
    original_data_path = os.path.join(opt["datapath"], "facebook_msc_original")
    check_original_data(original_data_path)
    return original_data_path


# if __name__ == '__main__':
#     # session 1
#     _root_save_path = os.path.join("/home/zhangtong/ParlAI/data/", 'facebook_msc')
#     _session1_root_save_path = os.path.join(_root_save_path, "session1")
#     _original_data_path = os.path.join("/home/zhangtong/ParlAI/data/", "facebook_msc_original")
#     reformat_session1_data(_original_data_path, _session1_root_save_path, "both_original")
