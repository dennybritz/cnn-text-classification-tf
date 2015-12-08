# CNN for Text Classification in Tensorflow

This is a simpliefied implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow. 

Notable differences are:

- No pretrained word vectors or dual channels. However, it's easy to extend the model to use word2vec.
- No weight clipping for regularization, only dropout.

## TODO

- Refactor code into init function
- Refactor model params into options object
- Use built-in logit functions
- Add L2 Loss?
- Remove affine dim
- Remove word2vec code

## Requirements

- Python 2.7
- Tensorflow

## Setup

TODO

## References

TOOD