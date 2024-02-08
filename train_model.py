from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task="twitter_conv",
        model="hred",
        model_file="tmp/hred2/hred_twitter_conv.model",
        dict_tokenizer="split",
        embedding_type="random",
        batchsize=64,
        truncate=128,
        metrics="all, gpu_mem",
        tensorboard_log=True,
        validation_metric='ppl',
        label_truncate=30,
        validation_metric_mode='min',
        # validation_every_n_secs=300,
        validation_every_n_epochs=1,
        validation_patience=5,
        skip_generation=True,
    )
    # parser.set_defaults(
    #     task="twitter_conv",
    #     model="hromc",
    #     model_file="tmp/hromc/hromc_basic.model",
    #     dict_tokenizer="split",
    #     embedding_type="random",
    #     batchsize=8,
    #     truncate=128,
    #     label_truncate=30,
    #     # metrics="rouge, token_acc, accuracy, bleu, f1, ppl, gpu_mem",
    #     metrics="all, gpu_mem",
    #     tensorboard_log=True,
    #     validation_metric='ppl',
    #     validation_metric_mode='min',
    #     # validation_every_n_secs=300,
    #     validation_every_n_epochs=1,
    #     validation_patience=5,
    #     skip_generation=True,
    # )
    opt = parser.parse_args()
    TrainLoop(opt).train()