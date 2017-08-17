# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


class PetaMultilabelDataLayerSync(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        
        self.TARGET_LABELS = ('hairLong', 'personMale', 'carryingBackpack','accessoryHat',\
                              'carryingOther','personalLess30', 'upperBodyVNeck')
        
        self.batch_loader = BatchLoader(params, None, self.TARGET_LABELS)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(self.batch_size, len(self.TARGET_LABELS ))

        print_info("PetaMultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result, TARGET_LABELS):
        self.result = result
        self.batch_size = params['batch_size']
        self.peta_root = params['peta_root']
        self.im_shape = params['im_shape']                
        self.TARGET_LABELS = TARGET_LABELS
        
        self.label_to_index = {label.lower(): i  for i, label in enumerate(self.TARGET_LABELS)}
        # get list of image indexes.
        list_file = params['split'] + '.txt'        
        self.annotations = load_peta_annotation(osp.join(self.peta_root, list_file))
        
        self.indexlist = self.annotations.keys()      
        
        
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = index 
        im = np.asarray(Image.open(
            osp.join(self.peta_root,  image_file_name)))
        im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        # Load and prepare ground truth
        multilabel = np.zeros(len(self.TARGET_LABELS)).astype(np.float32)
        labels = self.annotations[index]
        for label in labels:
            # in the multilabel problem we don't care how MANY instances
            # there are of  brand new worldeach class. Only if they are present.
            # The "-1" is b/c we are not interested in the background
            # class.
            label = label.lower()
            if label in self.label_to_index:
                label_index = self.label_to_index[label]
                multilabel[label_index] = 1

        self._cur += 1
        return self.transformer.preprocess(im), multilabel


def load_peta_annotation(list_file):
    
    annotations = {}
   
    with open(list_file) as f:
        for line in f:
            terms = line.strip("\n").split(" ")
            img_path = terms[0]
            labels = terms[1:]
            annotations[img_path] = labels
            
    return annotations        



def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'peta_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
