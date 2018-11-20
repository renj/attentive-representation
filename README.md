# Attentive Representation Learning


This project is the implementation of the paper [Same Representation, Different Attentions: Shareable Sentence  Representation Learning from Multiple Tasks](https://arxiv.org/pdf/1804.08139.pdf) (IJCAI 2018)

## Requirements

- python2.7
- pytorch
- sacred
- torchtext 0.1.1

Please download `glove.6B.100d.txt` and put into `embed` folder.


## Training

```
# Train simple classifier with attention using sentiment dataset on gpu 0
python main.py with data_set='product/sentiment' use_model='SingleAttn' device=0 adversarial=False ex_name='experiment'
```

- Arguments
    - use_model
        + model architecture used for training

To be continued ...