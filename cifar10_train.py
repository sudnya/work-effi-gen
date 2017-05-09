# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import numpy as np
import os

from datetime import datetime
import time

import tensorflow as tf

from model import Model

logger = logging.getLogger("Train")


#INITIAL_LEARNING_RATE = 1.0e-2       # Initial learning rate.


def train(model, config):
    
    """Train CIFAR-10 for a number of steps."""
    batch_size  = config.get("batch_size")
    output_path = config.get("output_path")

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        logger.info("Process inputs and labels for train/val sets")

        images, labels         = model.process_inputs(is_training=True)
        dev_images, dev_labels = model.process_inputs(is_training=False)

        is_validating = tf.placeholder(dtype=bool,shape=())
        
        images = tf.cond(is_validating, lambda:dev_images, lambda:images)
        labels = tf.cond(is_validating, lambda:dev_labels, lambda:labels)

        # Build a Graph that computes logits predictions from the inference model
        logits = model.inference(images)

        # Calculate loss.
        logger.info("Calculate loss")
        loss = model.loss(logits, labels)
        
        accuracy = model.accuracy(logits, labels)

        # calculate predictions
        logger.info("Top K predictions")
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Optimizer minimizes loss.
        logger.info("Run optimizer to minimize loss")
        #optimizer = tf.train.AdamOptimizer(learning_rate=INITIAL_LEARNING_RATE).minimize(loss)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.get("learning_rate")).minimize(loss)
        optimizer = tf.train.MomentumOptimizer(learning_rate=config.get("learning_rate"), momentum=config.get("momentum")).minimize(loss)

        # Initializer
        initializer = tf.global_variables_initializer()

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        validation_loss = tf.summary.scalar("validation loss", loss)
        validation_accuracy = tf.summary.scalar("validation accuracy", accuracy)

        display_step    = config.get("display_step")
        epochs          = config.get("epochs")
        steps_per_epoch = config.get("steps_per_epoch")

        iteration = 1
        validation_iteration = 1
        expcost = None

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(initializer)
            saver.save(sess, os.path.join(output_path, 'model'))

        for epoch in range(epochs):
            logger.info("Epoch: " + str(epoch))
            with tf.Session() as sess:
                epoch_saver = tf.train.import_meta_graph(os.path.join(output_path, 'model.meta'))
                epoch_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(output_path, './')))

                summary_writer = tf.summary.FileWriter(output_path, graph=tf.get_default_graph())

                # Training
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                logger.debug("Training epoch: " + str(epoch))
                step = 1
                try:
                    while not coord.should_stop() and step < steps_per_epoch:
                        _, cost, acc, summary = sess.run([optimizer, loss, accuracy, merged_summary_op], feed_dict={is_validating: False})
                        
                        if expcost == None:
                            expcost = cost
                        else:
                            expcost = (0.01) * cost + 0.99*expcost

                        summary_writer.add_summary(summary, iteration)

                        if step % display_step == 0:
                            logger.info("Iteration " + str(step*batch_size) + ", minibatch Loss= {:.6f}".format(cost) + " moving avg loss {:.6f}".format(expcost) + " accuracy: {:.4f}".format(acc))

                        step += 1
                        iteration += 1

                except tf.errors.OutOfRangeError:
                    logger.info('End of epoch')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                saver.save(sess, os.path.join(output_path, 'model'))

            with tf.Session() as sess:
                epoch_saver = tf.train.import_meta_graph(os.path.join(output_path, 'model.meta'))
                epoch_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(output_path, './')))
                # Eval
                logger.info("Evaluation")
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                step = 1
                try:
                    while not coord.should_stop() and step < steps_per_epoch:
                        _, _, loss_summary, accuracy_summary = sess.run([loss, accuracy, validation_loss, validation_accuracy], feed_dict={is_validating: True})
                        summary_writer.add_summary(loss_summary, validation_iteration)
                        summary_writer.add_summary(accuracy_summary, validation_iteration)
                        step += 1
                        validation_iteration += 1
                except tf.errors.OutOfRangeError:
                    logger.info('End of epoch')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()
                    
                coord.join(threads)


# one spot to set defaults
def initialize_defaults(config):
    if not config.get("data_dir"):
        config["data_dir"] = "/tmp/cifar10_data"
        logger.debug("data_dir not set in config file, default to " + str(config.get("data_dir")))
    if not config.get("output_path"):
        logger.debug("output_path not set in config file, default to " + str(config.get("output_path")))
        config["output_path"] = "/tmp/cifar10_output"
    if not config.get("epochs"):
        config["epochs"] = 2
        logger.debug("epochs not set in config file, default to  " + str(config.get("epochs")))
    if not config.get("batch_size"):
        config["batch_size"] = 32
        logger.debug("batch_size not set in config file, default to  " + str(config.get("batch_size")))
    if not config.get("learning_rate"):
        config["learning_rate"] = 1e-5
        logger.debug("learning_rate not set in config file, default to  " + str(config.get("learning_rate")))
    if not config.get("momentum"):
        config["momentum"] = 0.95
        logger.debug("momentum not set in config file, default to  " + str(config.get("momentum")))
    if not config.get("display_step"):
        config["display_step"] = 20 
        logger.debug("display_step not set in config file, default to  " + str(config.get("display_step")))
    if not config.get("steps_per_epoch"):
        config["steps_per_epoch"] = 100
        logger.debug("steps_per_epoch not set in config file, default to  " + str(config.get("steps_per_epoch")))


def main(argv=None):  # pylint: disable=unused-argument
    parser = argparse.ArgumentParser(description="Training driver")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("-c", "--config_file", default="config.json")

    parsed_arguments = parser.parse_args()
    arguments = vars(parsed_arguments)

    is_verbose   = arguments['verbose']
    config_file  = arguments['config_file']

    if is_verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    with open(config_file) as fid:
        config = json.load(fid)
        initialize_defaults(config)


    data_dir    = config.get("data_dir")
    output_path = config.get("output_path")
    model       = Model(config, is_verbose)
    model.maybe_download_and_extract(data_dir)

    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)
    
    train(model, config)


if __name__ == '__main__':
    tf.app.run()
