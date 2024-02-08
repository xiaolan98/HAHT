from parlai.scripts.eval_model import setup_args, eval_model
from parlai.core.agents import create_agent
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task="facebook_msc",
        model="memnn_msc",
        # model_file="../../data/tmp/session_opening/haht/facebook_msc.model"
        model_file="../../data/tmp/all_session/haht/facebook_msc.model",

        include_session=5,
        dict_lower=True,
        # session_opening_only=True,
        add_user_token=True,
        excluding_session1=True,
        # msc_passage_type="whole",
        msc_passage_type="separate",

        decoder_memory_attention=False,
        memory_module_type="transformer",
        # memory_module_type="none",
        hist_aware_cxt=True,
        copy_net=True,
        init_by_blender=True,
        share_hist_cont_encoder=False,
        reduction_type="max",

        batchsize=32,
        fp16=True,
        text_truncate=256,
        log_every_n_secs=60,
        # text_truncate=512,
        label_truncate=128,
        metrics="all, gpu_mem",
        tensorboard_log=True,
        datatype='test',
    )
    opt = parser.parse_args()
    eval_model(opt)
