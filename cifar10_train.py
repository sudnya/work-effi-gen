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

import cifar10

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train', """Directory where to write event logs """ """and checkpoint.""")
#tf.app.flags.DEFINE_integer('max_steps', 10000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
#tf.app.flags.DEFINE_integer('log_frequency', 100, """How often to log results to the console.""")

INITIAL_LEARNING_RATE = 1.0e-2       # Initial learning rate.


def train(config):
    cifar10.config = config

    """Train CIFAR-10 for a number of steps."""
    batch_size = config.get("batch_size", 32)
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = cifar10.process_inputs(is_training=True)
        dev_images, dev_labels = cifar10.process_inputs(is_training=False)

        is_validating = tf.placeholder(dtype=bool,shape=())
        
        images = tf.cond(is_validating, lambda:dev_images, lambda:images)
        labels = tf.cond(is_validating, lambda:dev_labels, lambda:labels)

        # Build a Graph that computes logits predictions from the inference model
        logits = cifar10.inference(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)
        
        # calculate predictions
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Optimizer minimizes loss.
        #optimizer = tf.train.AdamOptimizer(learning_rate=INITIAL_LEARNING_RATE).minimize(loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=INITIAL_LEARNING_RATE).minimize(loss)

        # Initializer
        initializer = tf.global_variables_initializer()

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        display_step = config.get("display_step", 20) 
        epochs = config.get("epochs", 20)
        steps_per_epoch = config.get("steps_per_epoch", 100)

        iteration = 1
        expcost = None

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(initializer)
            saver.save(sess, 'my-model')

        for epoch in range(epochs):
            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph('my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint('./'))

                summary_writer = tf.summary.FileWriter(config.get("train_dir", ""), graph=tf.get_default_graph())

                # Training
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                print("Epoch " + str(epoch))
                step = 1
                try:
                    while not coord.should_stop() and step < steps_per_epoch:
                        _, cost, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={is_validating: False})
                        
                        if expcost == None:
                            expcost = cost
                        else:
                            expcost = (0.01) * cost + 0.99*expcost

                        summary_writer.add_summary(summary, iteration)

                        if step % display_step == 0:
                            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                                    "{:.6f}".format(cost) + " {:.6f}".format(expcost))

                        step += 1
                        iteration += 1

                except tf.errors.OutOfRangeError:
                    print('End of epoch')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                saver.save(sess, 'my-model')

            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph('my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint('./'))
                # Eval
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                true_count = 0
                total_sample_count = 0
                step = 1
                try:
                    while not coord.should_stop() and step < steps_per_epoch:
                        predictions = sess.run([top_k_op], feed_dict={is_validating: True})
                        true_count += np.sum(predictions)
                        total_sample_count += batch_size
                        step += 1
                except tf.errors.OutOfRangeError:
                    print('End of epoch')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()
                    
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                summary = tf.Summary()
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, iteration)


#        class _LoggerHook(tf.train.SessionRunHook):
#          """Logs loss and runtime."""
#
#          def begin(self):
#              self._step = -1
#              self._start_time = time.time()
#
#          def before_run(self, run_context):
#              self._step += 1
#              return tf.train.SessionRunArgs(loss)  # Asks for loss value.
#
#          def after_run(self, run_context, run_values):
#              if self._step % config.get("log_frequency", 100) == 0:
#                  current_time = time.time()
#                  duration = current_time - self._start_time
#                  self._start_time = current_time
#
#                  loss_value = run_values.results
#                  examples_per_sec = config.get("log_frequency", 100) * config.get("batch_size", 32) / duration
#                  sec_per_batch = float(duration / config.get("log_frequency", 100))
#
#                  format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
#                                'sec/batch)')
#                  print (format_str % (datetime.now(), self._step, loss_value,
#                                     examples_per_sec, sec_per_batch))
#
#        with tf.train.MonitoredTrainingSession(
#            checkpoint_dir=config.get("train_dir" "/tmp"),
#            hooks=[tf.train.StopAtStepHook(last_step=config.get("max_steps", 10000)),
#            tf.train.NanTensorHook(loss),
#            _LoggerHook()],
#            config=tf.ConfigProto(
#            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
#            while not mon_sess.should_stop():
#                mon_sess.run(train_op)


def run_epoch(model, data_loader, session, summarizer):
    summary_op = tf.summary.merge_all()

    for batch in data_loader.train:
        ops = [model.train_op, model.avg_loss,
               model.avg_acc, model.it, summary_op]

        res = session.run(ops, feed_dict=model.feed_dict(*batch))
        _, loss, acc, it, summary = res
        summarizer.add_summary(summary, global_step=it)
        if it == 50:
            model.set_momentum(session)
            logger.debug("Setting initial momentum in iteration " + str(it))

        msg = "Iter {}: AvgLoss {:.3f}, AvgAcc {:.3f}"
        logger.debug(msg.format(it, loss, acc))
        if it % 100 == 0:
            msg = "Iter {}: AvgLoss {:.3f}, AvgAcc {:.3f}"
            logger.info(msg.format(it, loss, acc))
    return acc

def run_training(epochs):
    
    for e in range(epochs):
            start = time.time()
            train_acc = run_epoch(model, data_loader, sess, summarizer)
            saver.save(sess, os.path.join(output_save_path, "model"))
            logger.info("Epoch {} time {:.1f} (s)".format(e, time.time() - start))
            eval_acc = run_validation(model, data_loader, sess, summarizer)

            if eval_acc > best_eval_acc:
                saver.save(sess, os.path.join(output_save_path, "best_model.epoch"))
                best_eval_acc = eval_acc
                logger.info("Best accuracy so far: " + str(best_eval_acc))


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

    data_dir = config.get("data_dir", "/tmp/cifar10_data")
    train_dir = config.get("train_dir", "/tmp/cifar10_train")
    epochs = config.get("epochs", 1000)

    cifar10.maybe_download_and_extract(data_dir)

    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    train(config)
    #run_training()


if __name__ == '__main__':
    tf.app.run()
