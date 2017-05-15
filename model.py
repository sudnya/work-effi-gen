import logging
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import data_loader

logger = logging.getLogger("model")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
#INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

class Model:
    def __init__(self, is_verbose):
        if is_verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        

    def construct_model(self, config):
        """Create a NN model based on config
        Add all the variables to be initialized, and the variables that will 
        change
        Args:
            config: specifies model params
        Returns:
            Nothing
        """
        # [NHWC] - (batch_size, height, width, channels)
        self.inputs = tf.placeholder(tf.float32,
                      shape=(config.get("batch_size"), 
                          config.get("input_height"), 
                          config.get("input_width"), 
                          config.get("input_channels")))

        acts = self._build_nn(config.get("model"), self.inputs)

        output_dims = (config.get("batch_size"), 
                       config.get("output_classes"))
        logits = tf.contrib.layers.fully_connected(acts, output_dims,
                                                   activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(seed=2017))
        self.probs = tf.nn.softmax(logits)

        labels = tf.placeholder(tf.int64, shape=(self.batch_size))
        self.loss      = self._get_loss(logits, labels)
        self.accuracy  = self._get_accuracy(logits, labels)
        self.optimizer = self._get_optimizer(config)
        self.train_op = optimizer.minimize(self.loss)


    def get_train_step_op(self):
        return self.train_op

    def get_accuracy(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_inputs(self):
        return self.inputs

    def _get_loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]

        Returns:
            Loss tensor of type float.
        """
        labels = tf.cast(labels, tf.int64)
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name='cross_entropy_per_example')
        
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss


    def _get_accuracy(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        correct = tf.equal(tf.argmax(logits, dimension=1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy
    


    def _build_nn(self, model_cfg, inputs):
        #if model_cfg.get("name", "cnn") == "cnn":
        return self._build_cnn(model_cfg, inputs)
        #if model_cfg.get("name") == "rnn":
        #    return _build_rnn(model_cfg, inputs)
        #else:
        #    raise ValueError("Currently nn: " + str(model_cfg.get("name")) 
        #            + " is not supported")

    def _build_cnn(self, cfg, inputs):
        config = cfg.get("cnn").get("layer_config")
        acts = inputs

        ctr = 0
        for layer in config['conv_layers']:
            num_filters = layer['num_filters']
            filter_size = layer['filter_size']
            stride      = layer['stride']
            bn          = layer.get('enable_batch_norm', None)
            ln          = layer.get('enable_layer_norm', None)

            if bn is not None or ln is not None:
                acts = tf.contrib.layers.convolution2d(acts, 
                                                       num_outputs=num_filters,
                                                       kernel_size=[filter_size, 1],
                                                       stride=stride,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=ctr),
                                                       biases_initializer=None,
                                                       activation_fn=None)
                logger.debug ("Next activation mat shape " + str(acts.shape))

                if bn == True:
                    logger.debug("Adding Batch Norm Layer")
                    acts = tf.contrib.layers.batch_norm(acts, decay=0.9, 
                                                        center=True, scale=True, 
                                                        epsilon=1e-8,
                                                        activation_fn=tf.nn.relu,
                                                        is_training=True)

                elif ln == True:
                    logger.debug("Adding Layer Norm Layer")
                    acts = tf.contrib.layers.layer_norm(acts, center=True,
                                                        scale=True,
                                                        activation_fn=tf.nn.relu)
                else:
                    assert True, "Batch or Layer norm must be specified as True"
            else:
                acts = tf.contrib.layers.convolution2d(acts, 
                                                       num_outputs=num_filters,
                                                       kernel_size=[filter_size, 1],
                                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=ctr),

                                                       stride=stride)
                logger.debug ("Next activation mat shape " + str(acts.shape))
            ctr += 1

        return acts



    def _build_rnn(self, cfg, inputs):
        acts = self.inputs

        config = cfg.get("rnn").get("layer_config")
        rnn_dim = config['dim']
        cell_type = config.get('cell_type', 'gru')
        
        if cell_type == 'gru':
            logger.info("Adding cell type " + cell_type + " to rnn")
            cell = tf.contrib.rnn.GRUCell(rnn_dim)
        elif cell_type == 'lstm':
            logger.info("Adding cell type " + cell_type + " to rnn")
            cell = tf.contrib.rnn.LSTMCell(rnn_dim)
        else:
            msg = "Invalid cell type {}".format(cell_type)
            raise ValueError(msg)

        acts, _ = tf.nn.dynamic_rnn(cell, acts, dtype=tf.float32, 
                                        scope=None)
        return acts



    def _get_optimizer(self, config):
        return tf.train.GradientDescentOptimizer(config.get('learning_rate'))



    def maybe_download_and_extract(self, data_dir):
        """Download and extract the tarball from Alex's website."""
        dest_directory = data_dir

        if not os.path.exists(dest_directory):
            print ("Download and extract the tarball from Alex's website.")
            logger.info ("Creating directory: " + str(dest_directory))
            os.makedirs(dest_directory)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, 
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            
            filepath, _ = urllib.request.urlretrieve(DATA_URL, 
                                                    filepath, 
                                                    _progress)
            statinfo = os.stat(filepath)
            logger.info('Successfully downloaded' + str(filename) + 
                        str(statinfo.st_size) + 'bytes.')
        
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
        
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

