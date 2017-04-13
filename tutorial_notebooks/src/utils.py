"""
Helper functions/classes for reading in and representing data and
running experiments.

Some functions based on `mnist.py` from the `TensorFlow` tutorials.
"""
import re
from glob import glob
from os import listdir
from itertools import chain
from os.path import join, isdir, basename

import numpy as np
import pandas as pd
import tensorflow as tf

# Regex for replacing non-alphanumeric characters (not including hyphens
# and apostrophes)
NON_ALPHA_RE = re.compile(r"[^a-z0-9\-']+")


def read_data(data_path,
              macro_or_micro="macro",
              dev_set=True):
    """
    Read in data from a directory with the following files:
    testing_macro.csv and training_macro.csv or
    testing_micro/training_micro.csv.

    Either read in the "macro" files or the "micro" files.

    Returns 1) training IDs, features, and labels,
            2) test IDs, features, and labels, and
            3) development set IDs, features, and labels (if `dev_set` is
               False, the test test set will contain all data originally in
               the "testing" file).
    """

    train_data_path = join(data_path,
                           "training_{}.csv"
                           .format(macro_or_micro if macro_or_micro == "macro"
                                   else "micro_revised"))
    train_data = pd.read_csv(train_data_path, dtype={'appointment_id': str})
    train_labels = train_data['H1'].apply(lambda x: x - 1)
    train_features = train_data[[a for a in train_data.columns
                                 if a not in ['appointment_id', 'H1']]]
    train_ids = train_data['appointment_id']

    test_data_path = join(data_path,
                          "testing_{}.csv"
                          .format(macro_or_micro if macro_or_micro == "macro"
                                  else "micro_revised"))
    test_data = pd.read_csv(test_data_path, dtype={'appointment_id': str})

    test_labels = test_data['H1'].apply(lambda x: x - 1)
    test_features = test_data[[a for a in test_data.columns
                               if a not in ['appointment_id', 'H1']]]
    test_ids = test_data['appointment_id']

    dev_features = None
    dev_labels = None
    dev_ids = None
    if dev_set:
        test_amount = len(test_features) - 1000
        dev_features = test_features.head(1000)
        dev_features.index = range(len(dev_features))
        test_features = test_features.tail(test_amount)
        test_features.index = range(len(test_features))
        dev_labels = test_labels.head(1000)
        dev_labels.index = range(len(dev_labels))
        test_labels = test_labels.tail(test_amount)
        test_labels.index = range(len(test_labels))
        dev_ids = test_ids[:1000]
        test_ids = test_ids[1000:]

    return (train_ids,
            train_features,
            train_labels,
            test_ids,
            test_features,
            test_labels,
            dev_ids,
            dev_features,
            dev_labels)


def read_text_files_and_labels(labels_dict,
                               train_data_path,
                               test_data_path,
                               dev_data_path=None,
                               get_id_from_text_file_func=lambda x: x,
                               random_=False):
    """
    Read in text files and vectorize their contents so that each text
    is represented by a vector of the same size and also map them to
    labels and IDs. Return Dataset objects for training, test, and
    development sets (development set will be None if `dev_data_path`
    is None).

    :param labels_dict: dictionary mapping IDs to labels
    :type labels_dict: must contain a label/ID pair for every ID
                       in the training/test/dev data
    :param train_data_path: glob-style pattern for training data file
                            paths
    :type train_data_path: str
    :param test_data_path: glob-style pattern for test data file
                           paths
    :type test_data_path: str
    :param dev_data_path: glob-style pattern for dev data file paths
                          (None if no development set)
    :type dev_data_path: str or None
    :param get_id_from_text_file_func: function to use to extract IDs
                                       from text file names
    :type get_id_from_text_file_func: function (by default, this
                                      function simply strips off the
                                      ".txt" extension)
    :param random_: value for random_ parameter passed into Dataset
                    object, used for shuffling data
    :type random_: bool

    :returns: training, test, and development set DataSets
              (development set DataSet might be None if no args were
              provided to initalize it)
    :rtype: (Dataset, Dataset, Dataset or None)
    """

    data_paths = [train_data_path, test_data_path]
    partitions = ["training", "test"]
    word_lists_training = []
    train_ids = []
    train_labels = []
    word_lists_test = []
    test_ids = []
    test_labels = []
    word_lists_list = [word_lists_training, word_lists_test]
    ids_lists = [train_ids, test_ids]
    labels_lists = [train_labels, test_labels]
    if dev_data_path:
        data_paths.append(dev_data_path)
        partitions.append("dev")
        word_lists_dev = []
        dev_ids = []
        dev_labels = []
        word_lists_list.append(word_lists_dev)
        ids_lists.append(dev_ids)
        labels_lists.append(dev_labels)
    for (word_lists,
         ids_list,
         labels_list,
         data_path) in zip(word_lists_list,
                           ids_lists,
                           labels_lists,
                           data_paths):
        file_paths = glob(data_path)
        if not file_paths:
            raise ValueError("glob('{}') resulted in no matching file paths!"
                             .format(data_path))
        for file_path in file_paths:
            id_ = get_id_from_text_file_func(basename(file_path))
            ids_list.append(id_)
            labels_list.append(labels_dict[id_])
            with open(file_path) as text_file:
                word_lists.append(NON_ALPHA_RE.sub(text_file.read().strip().lower(),
                                                   r" ").split())

    # Get all unique words
    words = set()
    for word_list in chain(*word_lists_list):
        words.update(word_list)

    # Assign a unique number to each word
    words_vectorized = {w: i for i, w in enumerate(words)}

    # Get the maximum number of words across all text files
    word_list_lengths = []
    for (word_lists, partition) in zip(word_lists_list, partitions):
        word_list_lengths.extend([len(word_list) for word_list in word_lists])
    vector_size = np.max(word_list_lengths)

    padding_value = -1
    word_vectors_training = []
    word_vectors_test = []
    word_vectors_list = [word_vectors_training, word_vectors_test]
    if dev_data_path:
        word_vectors_dev = [word_vectors_dev]
        word_vectors_list.append()
    for (word_lists, word_vectors) in zip(word_lists_list, word_vectors_list):
        for word_list in word_lists:
            word_list_vectorized = [words_vectorized[w] for w in word_list]
            for i in range(len(word_list_vectorized), vector_size + 1):
                word_list_vectorized.append(padding_value)
            word_vectors.append(np.array(word_list_vectorized, dtype=np.int32))

    train_features = np.array(word_vectors_training, dtype=np.int32)
    if len(train_ids) != len(np.array(train_ids, dtype=np.int32)):
        raise ValueError("Decrease in precision causes ID duplicates.")
    train_ids = np.array(train_ids, dtype=np.int32)
    train_labels = np.array(train_labels, dtype=np.int32)
    if np.min(train_labels) == 1:
        train_labels = train_labels - 1
    test_features = np.array(word_vectors_test, dtype=np.int32)
    if len(test_ids) != len(np.array(test_ids, dtype=np.int32)):
        raise ValueError("Decrease in precision causes ID duplicates.")
    test_ids = np.array(test_ids, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)
    if np.min(test_labels) == 1:
        test_labels = test_labels - 1
    if dev_data_path:
        dev_features = np.array(word_vectors_dev, dtype=np.int32)
        if len(dev_ids) != len(np.array(dev_ids, dtype=np.int32)):
            raise ValueError("Decrease in precision causes ID duplicates.")
        dev_ids = np.array(dev_ids, dtype=np.int32)
        dev_labels = np.array(dev_labels, dtype=np.int32)
        if np.min(dev_labels) == 1:
            dev_labels = dev_labels - 1
    else:
        dev_features = None
        dev_ids = None
        dev_labels = None

    return (DataSet(train_ids, train_features, train_labels, random_=random_),
            DataSet(test_ids, test_features, test_labels, random_=random_),
            DataSet(dev_ids, dev_features, dev_labels, random_=random_) if dev_data_path else None)


class DataSet:

    def __init__(self, ids, features, labels, random_=False):
        prng = np.random.RandomState(12345)
        self._ids = ids
        self._features = features
        self._labels = labels
        self._index_in_epoch = 0
        self._num_examples = len(self._ids)
        if random_:
            reindex = prng.permutation(self._features.index)
            self._features = self._features.reindex(reindex)
            self._ids = self._ids.reindex(reindex)
            self._labels = self._labels.reindex(reindex)

    def get_size(self):
        return self._num_examples

    def next_batch(self, batch_size):

        start = self._index_in_epoch

        # Go to the next epoch
        if start + batch_size > self._num_examples:

            # Get the rest examples in this epoch
            remaining_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            ids_rest_part = self._ids[start: self._num_examples]


            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - remaining_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            ids_new_part = self._ids[start:end]
            labels_new_part = self._labels[start:end]
            return (np.concatenate((ids_rest_part, ids_new_part),
                                   axis=0),
                    np.concatenate((features_rest_part, features_new_part),
                                   axis = 0),
                    np.concatenate((labels_rest_part, labels_new_part),
                                   axis = 0))

        else:

            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return (self._ids[start:end],
                    self._features[start:end],
                    self._labels[start:end])


def inference(inputs, num_features, num_classes, hidden1_units,
              hidden2_units, hidden3_units=None):
    """
    Build a model on the inputs up to where it may be used for
    inference.

    Args:
        inputs: Placeholder for input data samples.
        num_features: Number of features in input data.
        num_classes: Number of classes/score labels.
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
        hidden3_units: Size of the third hidden layer (None if no
                       third layer).

    Returns:
        softmax_linear: Output tensor with the computed logits.
    """

    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
              tf.random_normal_initializer(0.0, 0.05)([num_features, hidden1_units]),
              name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)

    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
              tf.random_normal_initializer(0.0, 0.05)([hidden1_units, hidden2_units]),
              name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    if hidden3_units is not None:

        # Hidden 3
        with tf.name_scope('hidden3'):
            weights = tf.Variable(
                tf.random_normal_initializer(0.0, 0.05)([hidden2_units, hidden3_units],
                                    stddev=1.0 / math.sqrt(float(hidden2_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden3_units]),
                                 name='biases')
            hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

    # Linear
    with tf.name_scope('softmax_linear'):

        weights = tf.Variable(
              tf.random_normal_initializer(0.0, 0.05)([hidden2_units
                                                           if hidden3_units is None
                                                           else hidden3_units,
                                                       num_classes]),
              name='weights')
        biases = tf.Variable(tf.zeros([num_classes]),
                             name='biases')
        logits = tf.matmul(hidden2 if hidden3_units is None else hidden3,
                           weights) + biases

    return logits


def loss(logits, labels):
    """
    Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor of type float.
    """

    labels = tf.to_int64(labels)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')    


def training_adam(_loss, learning_rate):
    """
    Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an Adam optimizer and applies the gradients to all trainable
    variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        _loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', _loss)

    # Create the Adam optimizer with the given learning
    # rate.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single
    # training step.
    train_op = optimizer.minimize(_loss, global_step=global_step)

    return train_op


def training_gradient_descent(_loss, learning_rate):
    """
    Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.
    Creates a gradient descent optimizer and applies the gradients to
    all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        _loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', _loss)

    # Create the gradient descent optimizer with the given learning
    # rate.
    tf.train.GradientDescentOptimizer(learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single
    # training step.
    train_op = optimizer.minimize(_loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
                range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of
        batch_size) that were predicted correctly.
    """

    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)

    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def fill_feed_dict(data, inputs_pl, labels_pl, batch_size):
    """
    Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ...
    }

    Args:
        data_set: The set of features and labels.
        inputs_pl: The input data placeholder.
        labels_pl: The input labels placeholder.
        batch_size: Size of each batch.

    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """

    # Create the feed_dict for the placeholders filled with the next
    # `batch_size` samples.
    ids, inputs_feed, labels_feed = data.next_batch(batch_size)
    feed_dict = {
        inputs_pl: inputs_feed,
        labels_pl: labels_feed,
    }

    return feed_dict


def do_eval(sess, eval_correct, inputs_placeholder, labels_placeholder, data,
            logits, batch_size):
    """
    Runs one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct
                      predictions.
        inputs_placeholder: The input data placeholder.
        labels_placeholder: The labels placeholder.
        data: The set of data and labels to evaluate.
        logits: List of logits,
        batch_size: Size of each batch of data.
    """

    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data.get_size() // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data,
                                   inputs_placeholder,
                                   labels_placeholder,
                                   batch_size)
        feed_dict[keep_prob] = dropout
        logit_output, true_cnt = sess.run([logits, eval_correct],
                                          feed_dict=feed_dict)
        true_count += true_cnt

    acc = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %
          (num_examples, true_count, acc))


def do_eval_cnn(sess, eval_correct, inputs_placeholder, labels_placeholder,
                data, logits, batch_size, keep_prob, dropout):
    """
    Runs one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct
                      predictions.
        inputs_placeholder: The input data placeholder.
        labels_placeholder: The labels placeholder.
        data: The set of data and labels to evaluate.
        logits: List of logits,
        batch_size: Size of each batch of data.
    """

    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data.get_size() // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data,
                                   inputs_placeholder,
                                   labels_placeholder,
                                   batch_size)
        feed_dict[keep_prob] = dropout
        logit_output, true_cnt = sess.run([logits, eval_correct],
                                          feed_dict=feed_dict)
        true_count += true_cnt

    acc = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %
          (num_examples, true_count, acc))
