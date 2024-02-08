from parlai.scripts.train_model import TrainLoop, setup_args
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
# import torch
# torch.manual_seed(0)


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task="facebook_msc",
        model="transformer/generator",
        # datatype="test",
        # init_model="../data/models/blender/blender_3B/model",
        init_model="../../data/models/blender/blender_90M/model",
        model_file="../../data/tmp/all_session/transformer/facebook_msc_session_5_no_hist/facebook_msc.model",
        # model_file="../../data/tmp/session_only/transformer_no_hist/"
                   # "facebook_msc_session_1/facebook_msc.model",
                   # "facebook_msc_session_2/facebook_msc.model",
                   # "facebook_msc_session_3/facebook_msc.model",
                   # "facebook_msc_session_4/facebook_msc.model",
        dict_file="../../data/models/blender/blender_90M/model.dict",
        dict_lower=True,
        batchsize=64,
        include_session=5,
        # model_parallel=True,
        add_user_token=False,
        learningrate=1e-06,
        embedding_size=512,
        n_layers=8,
        ffn_size=2048,
        dropout=0.1,
        n_heads=16,
        learn_positional_embeddings=True,
        n_positions=512,
        variant="xlm",
        activation='gelu',
        fp16=True,
        text_truncate=256,
        # text_truncate=512,
        label_truncate=128,
        dict_tokenizer='bpe',
        lower=True,
        optimizer="adamax",
        lr_scheduler="reduceonplateau",
        gradient_clip=0.2,
        veps=0.25,
        betas=[0.9, 0.999],
        update_freq=1,
        attention_dropout=0.0,
        relu_dropout=0.0,
        skip_generation=True,
        metrics="all, gpu_mem",
        tensorboard_log=True,
        validation_metric='loss',
        validation_metric_mode='min',
        # validation_every_n_secs=30,
        validation_every_n_epochs=1,
        validation_patience=5,
        lr_scheduler_patience=1,

    )
    opt = parser.parse_args()
    TrainLoop(opt).train()

