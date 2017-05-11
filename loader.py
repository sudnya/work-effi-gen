import argparse
import collections
import glob
import logging
import numpy as np
import os
import pickle
import random
import scipy.io as sio
import json

logger = logging.getLogger("Loader")

class Loader:
    """
    Loader class for feeding data to the network. This class loads the training
    and validation data sets. Once the datasets are loaded, they can be batched
    and fed to the network. Example usage:

        ```
        data_path = <path_to_data>
        batch_size = 32
        ldr = Loader(data_path, batch_size)
        for batch in ldr.train:
            run_sgd_on(batch)
        ```

    This class is also responsible for normalizing the inputs.
    """

    def __init__(self, data_path, batch_size,
                 seed=None, augment=False):
        """
        :param data_path: path to the training and validation files
        :param batch_size: size of the minibatches to train on
        :param seed: seed the rng for shuffling data
        :param augment: set to true to augment the training data
        """
        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        if seed is not None:
            random.seed(seed)

        self.batch_size = batch_size
        self.augment = augment

        self._train, self._val = load_all_data(data_path)
        logger.info("Training set has " + str(len(self._train)) + " samples")
        logger.info("Validation set has " + str(len(self._val)) + " samples")

        self.compute_mean_std()
        self._train = [(self.normalize(img), l) for img, l in self._train]
        self._val = [(self.normalize(img), l) for img, l in self._val]

        label_counter = collections.Counter(l for _, l in self._train)

        classes = sorted([c for c, _ in label_counter.most_common()])
        self._int_to_class = dict(zip(range(len(classes)), classes))
        self._class_to_int = {c : i for i, c in self._int_to_class.items()}
        self.class_counts = [label_counter[c] for c in classes]

        self._train = self.batches(self._train)
        self._val = self.batches(self._val)

    def batches(self, data):
        """
        :param data: the raw dataset from e.g. `loader.train`
        :returns: Iterator to the minibatches. Each minibatch consists
                  of an (img, labels) pair. Each img row of the array stores a 
                  32x32 colour image represented as 1x3072 entries. 
                  The first 1024 entries contain the red channel values, 
                  the next 1024 the green, and the final 1024 the blue. 
                  The image is stored in row-major order, so that the first 32 
                  entries of the array are the red channel values of the first 
                  row of the image.  
                  the labels is a list of integer labels.
        """
        # Sort by length
        #data = sorted(data, key = lambda x: x[0].shape[0])

        inputs, labels = zip(*data)
        labels = [self._class_to_int[l] for l in labels]
        batch_size = self.batch_size
        data_size = len(labels)

        end = data_size - batch_size + 1
        batches = [(inputs[i:i + batch_size], labels[i:i + batch_size])
                   for i in range(0, end, batch_size)]
        random.shuffle(batches)

        logger.debug("Data set {" + str(data_size) + " samples}, batch size {" \
                + str(batch_size) + "} -> " + str(len(batches)) + " batches")
        return batches

    def normalize(self, example):
        """
        Normalizes a given example by the training mean and std.
        :param: example: 1D numpy array
        :return: normalized example
        """
        return (example - self.mean) / self.std

    def compute_mean_std(self):
        """
        Estimates the mean and std over the training set.
        """
        n_samples = sum(len(w) for w, _ in self._train)
        mean_sum = np.sum([np.sum(w) for w, _ in self._train])
        mean = mean_sum / n_samples

        var_sum = np.sum([np.sum((w - mean)**2) for w, _ in self._train])
        var = var_sum / n_samples

        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(var).astype(np.float32)

    @property
    def classes(self):
        return [self._int_to_class[i]
                for i in range(self.output_dim)]

    @property
    def output_dim(self):
        """ Returns number of output classes. """
        return len(self._int_to_class)

    @property
    def train(self):
        """ Returns the raw training set. """
        for imgPixs, labels in self._train:
            if self.augment:
                imgPixs = [transform(imgPix) for imgPix in imgPixs]
            yield (imgPixs, labels)

    @property
    def val(self):
        """ Returns the raw validation set. """
        return self._val

#    def load_preprocess(self, record_id):
#        imgPix = load_imgPix_mat(record_id + ".mat")
#        return self.normalize(imgPix)

    def int_to_class(self, label_int):
        """ Convert integer label to class label. """
        return self._int_to_class[label_int]

    def __getstate__(self):
        """
        For pickling.
        """
        return (self.mean,
                self.std,
                self._int_to_class,
                self._class_to_int,
                self.class_counts)

    def __setstate__(self, state):
        """
        For unpickling.
        """
        self.mean = state[0]
        self.std = state[1]
        self._int_to_class = state[2]
        self._class_to_int = state[3]
        self.class_counts = state[4]

def transform(imgPix):
    scale = random.uniform(0.1, 5.0)
    flip = random.choice([-1.0, 1.0])
    return  imgPix * flip * scale

def load_all_data(data_path):
    """
    Returns tuple of training and validation sets. Each set
    will contain a list of pairs of raw imgPix and the
    corresponding label.
    """
    #label_file = os.path.join(data_path, "REFERENCE-v2.csv")
    # Load record ids + labels
    files = []
    files.append(os.path.join(data_path, 'data_batch_1'))
    files.append(os.path.join(data_path, 'data_batch_2'))
    files.append(os.path.join(data_path, 'data_batch_3'))
    files.append(os.path.join(data_path, 'data_batch_4'))
    files.append(os.path.join(data_path, 'data_batch_5'))

    train_records = []
    test = []
    train_labels = []
    for f in files:
        logger.debug("Data set from file " + str(f))
        with open(f, 'rb') as fo:
            record_dict = pickle.load(fo, encoding='bytes')
            #for key in record_dict:
            #    print(key.decode("utf-8"))
            #d = record_dict.get("data".encode("utf-8"))
            #l = record_dict.get("labels".encode("utf-8"))
            #test.extend((d,l))
            train_records.extend(record_dict.get("data".encode("utf-8")))
            train_labels.extend(record_dict.get("labels".encode("utf-8")))

    val_f = os.path.join(data_path, "test_batch")
    val_records = []
    val_labels = []
    with open(val_f, 'rb') as fo:
        logger.debug("Data set from file " + str(f))
        record_dict = pickle.load(fo, encoding='bytes')
        val_records.extend(record_dict.get("data".encode("utf-8")))
        val_labels.extend(record_dict.get("labels".encode("utf-8")))

    train = []
    for data, label in zip(train_records, train_labels):
        train.append((data, label))

    val = []
    for data, label in zip(val_records, val_labels):
        val.append((data, label))


    return train, val

#def load_imgPix_mat(imgPix_file):
#    return sio.loadmat(imgPix_file)['val'].squeeze()

def main():
    parser = argparse.ArgumentParser(description="CIFAR10 Loader")
    parser.add_argument("-v", "--verbose",
            default = False, action = "store_true")
    parser.add_argument("-p", "--data_path",
            default="/deep/group/sudnya/cifar-10-batches-py")
    parser.add_argument("-b", "--batch_size", default=32)

    parsed_arguments = parser.parse_args()
    arguments = vars(parsed_arguments)

    is_verbose   = arguments['verbose']
    data_path    = arguments['data_path']
    batch_size   = arguments['batch_size']


    if is_verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    random.seed(2016)
    ldr = Loader(data_path, batch_size)
    logger.info("Length of training set {}".format(sum(1 for x in ldr.train)) 
            + " for batch size {}".format(batch_size))
    logger.info("Length of validation set {}".format(sum(1 for x in ldr.val)) 
            + " for batch size {}".format(batch_size))
    logger.info("Length of output classes {}".format(ldr.output_dim) 
            + " for batch size {}".format(batch_size))

    # Run a few sanity checks.
    count = 0
    for imgPixs, labels in ldr.train:
        count += 1
        assert len(imgPixs) == len(labels) == batch_size, "Invalid example count."
        assert len(imgPixs[0].shape) == 1, "ECG array should be 1D"

if __name__ == '__main__':
    main()

