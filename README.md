# Guiding Deep Learning System Testing using Surprise Adequacy

Code release of a paper ["Guiding Deep Learning System Testing using Surprise Adequacy"](https://arxiv.org/abs/1808.08444)

If you find this paper helpful, consider cite the paper:

```
@article{kim2018guiding,
  title={Guiding Deep Learning System Testing using Surprise Adequacy},
  author={Kim, Jinhan and Feldt, Robert and Yoo, Shin},
  journal={arXiv preprint arXiv:1808.08444},
  year={2018}
}
```

## Introduction

This repository includes [all experimental results](https://goo.gl/Dq63fq), code for computing Surprise Adequacy (SA).

- run.py - script processing SA with a benign dataset and adversarial examples (MNIST and CIFAR-10).
- sa.py - tools that fetch activation traces, compute LSA and DSA, and coverage.
- train_model.py - model training script for MNIST and CIFAR-10, keeping trained models in "model" directory (from [Ma et al.](https://github.com/xingjunm/lid_adversarial_subspace_detection)).
- model directory - saving models.
- tmp directory - saving activation traces and prediction arrays.
- adv directory - saving adversarial examples.

### Generating Adversarial Examples

We used a framework of [Ma et al.](https://github.com/xingjunm/lid_adversarial_subspace_detection) to generate various adversarial examples (FGSM, BIM-A, BIM-B, JSMA, and C&W). Please refer to [craft_adv_samples.py](https://github.com/xingjunm/lid_adversarial_subspace_detection/blob/master/craft_adv_examples.py) in above repository, and put them in "adv" directory.

### Udacity Self-driving Car Challenge

To reproduce the result of [Udacity self-driving car challenge](https://github.com/udacity/self-driving-car/tree/master/challenges/challenge-2), please refer to the [DeepXplore](https://github.com/peikexin9/deepxplore) and [DeepTest](https://github.com/ARiSE-Lab/deepTest) repositories, which contain information about dataset, models ([Dave-2](https://github.com/peikexin9/deepxplore/tree/master/Driving), [Chauffeur](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/chauffeur)), and synthetic data generation.

## How to Use

Our implementation is based on Python 3.5.2, Tensorflow 1.9.0, Keras 2.2, Numpy 1.14.5.

```bash
# install Python dependencies
pip install -r requirements.txt

# train a model
python train_model.py -d mnist

# calculate LSA, coverage, and ROC-AUC score
python run.py -lsa

# calculate DSA, coverage, and ROC-AUC score
python run.py -dsa
```

## Notes

- If you encounter "ValueError: Input contains NaN, infinity or a value too large for dtype
  ('float64')." error, you need to increase variance threshold. Please see the configuration details in the paper (Section IV-C).
- Images were processed by clipping its pixels in between -0.5 and 0.5.
- If you want to select some layers, you can modify layers array in run.py.
- Coverage may vary depending on the upper bound.

## References

- [DeepXplore](https://github.com/peikexin9/deepxplore)
- [DeepTest](https://github.com/ARiSE-Lab/deepTest)
- [Detecting Adversarial Samples from Artifacts](https://github.com/rfeinman/detecting-adversarial-samples)
- [Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality](https://github.com/xingjunm/lid_adversarial_subspace_detection)
