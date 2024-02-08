# HAHT


[History-Aware Hierarchical Transformer for Multi-session Open-domain Dialogue System](https://aclanthology.org/2022.findings-emnlp.247.pdf) <br>

Overview of HAHT
![HAHT Structure](./images/structure1.png)
![HAHT Structure](./images/structure2.png)


## Prerequisites
- Python 3.6
- PyTorch 1.10.0

## Getting Started

### Installation


```bash
python setup.py develop
```


### Training

To train the model, run:

```bash
cd debug_scripts/blender_haht/
python finetune_haht.py
```

To evaluate the model, run:

```bash
python evaluate_haht.py --skip_generation False --model_parallel False --batchsize 32
```

### Logging

TensorBoard logs and models will be saved in `HAHT/data/tmp/all_session/` folder.
Generate the responses and coresponding input conversation context will be saved in `HAHT/debug_scripts/blender_haht/logs/`

### Discussion

If you have difficulties to get things working in the above steps, please let us know.


