# Towards Better Selective Classification

This is the official implementation of the paper [Towards Better Selective Classification](https://arxiv.org/abs/2206.09034). 

In this work, we confirm that the superior performance of state-of-the-art methods such as [SelectiveNet](https://arxiv.org/abs/1901.09192), [Deep Gamblers](https://arxiv.org/abs/1907.00208), and [Self-Adaptive Training](https://arxiv.org/abs/2002.10319) is owed to training a more generalizable classifier rather than their proposed selection mechanisms. We propose an entropy-based regularizer that improves the performance and achieves new state-of-the-art results.

## Install

Create and activate a conda environment. Install the dependencies as listed in `requirements.txt`:

```
conda create --name sel_cls python=3.7
conda activate sel_cls
pip install -r requirements.txt
```

## Training and Evaluation

**Self-Adaptive Training (SAT):**
```
bash run_${dataset}.sh
```

**Self-Adaptive Training (SAT) + Entropy Minimization (EM):**
```
bash run_${dataset}_entropy.sh
```

## Reference

For technical details, please check the conference version of our paper.
```
@inproceedings{
    feng2023towards,
    title={Towards Better Selective Classification},
    author={Leo Feng and Mohamed Osama Ahmed and Hossein Hajimirsadeghi and Amir H. Abdi},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=5gDz_yTcst}
}
```

## Acknowledgement

This code is based on the official code base of [Self-Adaptive Training](https://github.com/LayneH/SAT-selective-cls) (which is based on  the official code base of [Deep Gambler](https://github.com/Z-T-WANG/NIPS2019DeepGamblers)).

