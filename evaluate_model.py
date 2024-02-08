from parlai.scripts.eval_model import setup_args, eval_model

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task="twitter_conv",
        model="hred",
        model_file="tmp/hred/hred_twitter_conv.model",
        dict_tokenizer="split",
        embedding_type="random",
        batchsize=64,
        truncate=128,
        label_truncate=30,
        metrics="all, gpu_mem",
        tensorboard_log=True,
        validation_metric='ppl',
        validation_metric_mode='min',
        # validation_every_n_secs=300,
        validation_every_n_epochs=1,
        validation_patience=5,
        datatype='test'
    )
    opt = parser.parse_args()
    eval_model(opt)