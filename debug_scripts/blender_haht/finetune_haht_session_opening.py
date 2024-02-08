from parlai.scripts.train_model import TrainLoop, setup_args
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task="facebook_msc",
        model="memnn_msc",
        init_model="../../data/models/blender/blender_90M/model",
        dict_file="../../data/models/blender/blender_90M/model.dict",
        model_file="../../data/tmp/session_opening/haht/facebook_msc.model",
        dict_lower=True,
        session_opening_only=True,
        add_user_token=True,
        excluding_session1=True,
        # msc_passage_type="whole",
        include_session=5,
        msc_passage_type="separate",
        # use_persona_as_history=True,

        decoder_memory_attention=False,
        memory_module_type="none",
        hist_aware_cxt=True,
        copy_net=True,
        init_by_blender=True,
        share_hist_cont_encoder=False,
        reduction_type="max",
        # glossaries=["_sep1_", "_sep2_"],
        # share_hist_cont_encoder=True,

        # model_parallel=True,
        model_parallel=False,
        batchsize=8,
        memory_hidden_size=512,
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
        # text_truncate=512,
        # text_truncate=128,
        text_truncate=256,
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
        validation_every_n_epochs=0.5,
        validation_patience=10,
        lr_scheduler_patience=2,

    )
    opt = parser.parse_args()
    TrainLoop(opt).train()