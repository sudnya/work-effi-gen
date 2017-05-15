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


def run_epoch(model, loader, session, summarizer, iteration):
    
    train_step = model.get_train_step_op()
    accuracy = model.get_accuracy()
    loss = model.get_loss()
    inputs = model.get_inputs()


    # create summary ops for things that you want to log
    tf.summary.scalar("training_loss", loss)
    tf.summary.scalar("training_accuracy", accuracy)
    
    # merge summaries for entire graph
    summary_op = tf.summary.merge_all()

    for batch in loader.train:
        var_list = [loss, acccuracy, train_step, summary_op]

        # update the model
        result = session.run(var_list, feed_dict={inputs : batch})

        result_loss, result_accuracy, _, result_summary = result
        
        # log summary
        summary_writer.add_summary(result_summary, iteration)

        iteration += 1

    return iteration


def run_validation(model, loader, session, summarizer, iteration):
    
    accuracy = model.get_accuracy()
    loss = model.get_loss()
    inputs = model.get_inputs()


    # create summary ops for things that you want to log
    tf.summary.scalar("validation_loss", loss)
    tf.summary.scalar("validation_accuracy", accuracy)
    
    # merge summaries for entire graph
    summary_op = tf.summary.merge_all()

    for batch in loader.val:
        var_list = [loss, acccuracy, summary_op]

        # update the model
        result = session.run(var_list, feed_dict={inputs : batch})

        result_loss, result_accuracy, _, result_summary = result
        
        # log summary
        summary_writer.add_summary(result_summary, iteration)

        iteration += 1

    return iteration


def run_training(model, config):
    output_save_path = config.get("output_path")
    
    tf.set_random_seed(config['seed'])
    
    # Variables that will be initialized, and the program itself
    model.construct_model(config)

    # Checkpointing
    saver = tf.train.Saver(tf.global_variables())

    initializer = tf.global_variables_initializers()

    with tf.Session() as sess:

        sess.run(initializer)

        summarizer = tf.summary.FileWriter(output_save_path, sess.graph)

        iteration = 0

        for epoch in range(config.get("epochs")):
            start = time.time()

            iteration = run_epoch(model, loader, sess, summarizer,
                                  iteration)

            saver.save(sess, os.path.join(output_save_path, "model"))

            eval_acc = run_validation(model, loader, sess, summarizer)


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
    model       = Model(is_verbose)
    model.maybe_download_and_extract(data_dir)

    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)
    
#    train(model, config)
    run_training(model, config)


if __name__ == '__main__':
    tf.app.run()

