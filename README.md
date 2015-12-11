# CNN for Text Classification in Tensorflow

A slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Requirements

- Python 2.7
- Tensorflow
- Numpy

## Running

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes
  --num_filters NUM_FILTERS
                        Number of filters, per filter size
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability
  --batch_size BATCH_SIZE
                        Batch Size
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow soft device placement (e.g. no GPU)
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement
```

Train:

```bash
./train.py
```

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification]()http://arxiv.org/abs/1510.03820