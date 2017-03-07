from os.path import join

import pandas as pd
import numpy as np
import tensorflow as tf


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
              tf.random_normal_initializer(0.0, 0.05)([hidden1_units, num_classes]),
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
        logit_output, true_cnt = sess.run([logits, eval_correct],
                                          feed_dict=feed_dict)
        true_count += true_cnt

    acc = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %
          (num_examples, true_count, acc))
