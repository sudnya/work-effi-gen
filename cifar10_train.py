import argparse
import json
import logging
import numpy as np
import os
import loader

from datetime import datetime
import time

import tensorflow as tf

from model import Model

logger = logging.getLogger("Train")


def run_training_epoch(model, loader, session, summarizer, iteration, 
                        summary_op):
    
    train_step = model.get_train_step_op()
    accuracy = model.get_accuracy()
    loss = model.get_loss()
    inputs = model.get_inputs()
    labels = model.get_labels()


    for batch in loader.train:
        #print (batch)
        var_list = [loss, accuracy, train_step, summary_op]

        # update the model
        batch_data = batch[0] 
        batch_labels = batch[1]

        result = session.run(var_list, feed_dict={
            inputs : batch_data,
            labels : batch_labels})

        result_loss, result_accuracy, _, result_summary = result
        logger.info("training loss " + str(result_loss) + " accuracy " +
                str(result_accuracy))
        
        # log summary
        summarizer.add_summary(result_summary, iteration)

        iteration += 1

    return iteration


def run_validation_epoch(model, loader, session, summarizer, iteration, 
                        summary_op):
    
    accuracy = model.get_accuracy()
    loss = model.get_loss()
    inputs = model.get_inputs()
    labels = model.get_labels()


    for batch in loader.val:
        var_list = [loss, accuracy, summary_op]

        batch_data = batch[0] 
        batch_labels = batch[1]
        # update the model
        result = session.run(var_list, feed_dict={
            inputs : batch_data,
            labels : batch_labels})

        result_loss, result_accuracy, result_summary = result
        logger.info("validation loss " + str(result_loss) + " accuracy " +
                str(result_accuracy))
        
        # log summary
        summarizer.add_summary(result_summary, iteration)

        iteration += 1

    return iteration


def run_training(model, loader, config):
    output_save_path = config.get("output_path")
    
    tf.set_random_seed(config.get('seed'))
    
    # Variables that will be initialized, and the program itself
    model.construct_model(config)

    # Checkpointing
    saver = tf.train.Saver(tf.global_variables())

    initializer = tf.initialize_all_variables()
    # create summary ops for things that you want to log
    accuracy = model.get_accuracy()
    loss = model.get_loss()
    
    training_loss_summary = tf.summary.scalar("training_loss", loss)
    training_accuracy_summary = tf.summary.scalar("training_accuracy", accuracy)
    
    validation_loss_summary = tf.summary.scalar("validation_loss", loss)
    validation_accuracy_summary = tf.summary.scalar("validation_accuracy", 
                                                    accuracy)
    
    # merge summaries for entire graph
    training_summary_op = tf.summary.merge([training_loss_summary, 
                                            training_accuracy_summary])
    validation_summary_op = tf.summary.merge([validation_loss_summary, 
                                            validation_accuracy_summary])


    with tf.Session() as sess:

        sess.run(initializer)

        summarizer = tf.summary.FileWriter(output_save_path, sess.graph)

        iteration = 0

        for epoch in range(config.get("epochs")):
            start = time.time()

            iteration = run_training_epoch(model, loader, sess, summarizer,
                                  iteration, training_summary_op)

            saver.save(sess, os.path.join(output_save_path, "model"))

            eval_acc = run_validation_epoch(model, loader, sess, summarizer, iteration,
                                      validation_summary_op)


# one stop shop to set defaults
def initialize_defaults(config):
    if not config.get("data_dir"):
        config["data_dir"] = "/tmp/cifar10_data"
        logger.debug("data_dir not set in config file, default to " 
                + str(config.get("data_dir")))
    if not config.get("output_path"):
        logger.debug("output_path not set in config file, default to " 
                + str(config.get("output_path")))
        config["output_path"] = "/tmp/cifar10_output"
    if not config.get("epochs"):
        config["epochs"] = 2
        logger.debug("epochs not set in config file, default to  " 
                + str(config.get("epochs")))
    if not config.get("batch_size"):
        config["batch_size"] = 32
        logger.debug("batch_size not set in config file, default to  " 
                + str(config.get("batch_size")))
    if not config.get("learning_rate"):
        config["learning_rate"] = 1e-5
        logger.debug("learning_rate not set in config file, default to  " 
                + str(config.get("learning_rate")))
    if not config.get("momentum"):
        config["momentum"] = 0.95
        logger.debug("momentum not set in config file, default to  " 
                + str(config.get("momentum")))
    if not config.get("display_step"):
        config["display_step"] = 20 
        logger.debug("display_step not set in config file, default to  " 
                + str(config.get("display_step")))
    if not config.get("steps_per_epoch"):
        config["steps_per_epoch"] = 100
        logger.debug("steps_per_epoch not set in config file, default to  " 
                + str(config.get("steps_per_epoch")))

    if not config.get("seed"):
        config["seed"] = 2017
        logger.debug("seed not set in config file, default to  " 
                + str(config.get("seed")))
    if not config.get("input_height"):
        config["input_height"] = 24
        logger.debug("input_height not set in config file, default to  " 
                + str(config.get("input_height")))
    if not config.get("input_width"):
        config["input_width"] = 24
        logger.debug("input_width not set in config file, default to  " 
                + str(config.get("input_width")))
    if not config.get("input_channels"):
        config["input_channels"] = 3
        logger.debug("input_channels not set in config file, default to  " 
                + str(config.get("input_channels")))

    if not config.get("output_classes"):
        config["output_classes"] = 10
        logger.debug("output_classes not set in config file, default to  " 
                + str(config.get("output_classes")))



def main(argv=None):
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
    ldr         = loader.Loader(data_dir, config)
    model       = Model(is_verbose)

    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)
    
    run_training(model, ldr, config)


if __name__ == '__main__':
    tf.app.run()

