"""Code for testing ADR."""
import json
import numpy as np
import os
import medpy.metric.binary as mmb
import random
import tensorflow as tf
import math


import model as model_output
from stats_func import *

import nibabel as nib

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '9'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


CHECKPOINT_PATH = '' #checkpoint path


BASE_FID = '' # folder path of test files
TESTFILE_FID = '' # path of the .txt file storing the test filenames
TEST_MODALITY = '' # MR or CT




KEEP_RATE = 1.0
IS_TRAINING = tf.cast(False,tf.bool)
BATCH_SIZE = 128

data_size = [256, 256, 1]
label_size = [256, 256, 1]

contour_map = {
    "bg": 0,
    "la_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4,
}




class ADR:

    def __init__(self, config):

        self.keep_rate = KEEP_RATE
        self.is_training = IS_TRAINING
        self.checkpoint_pth = CHECKPOINT_PATH
        self.batch_size = BATCH_SIZE

      
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])

        self.base_fd = BASE_FID
        self.test_fid = TESTFILE_FID
        
    def model_setup(self):

        self.input_a = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                1
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                1
            ], name="input_B")
        self.gt_a = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                self._num_cls
            ], name="gt_A")
        self.gt_b = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                self._num_cls
            ], name="gt_B")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
        }

        outputs = model_output.get_outputs(inputs, self.batch_size, skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)
        self.pred_real_b = outputs['pred_real_b']
      
        self.predicter_b = tf.nn.softmax(self.pred_real_b)
        self.compact_pred_b = tf.argmax(self.predicter_b, 3)
        self.compact_y_b = tf.argmax(self.gt_b, 3)

    def read_lists(self, fid):
        """read test file list """

        with open(fid, 'r') as fd:
            _list = fd.readlines()

        my_list = []
        for _item in _list:
            my_list.append(_item.split('\n')[0])
        return my_list

    def label_decomp(self, label_batch):
        """decompose label for one-hot encoding """

        _batch_shape = list(label_batch.shape)
        _vol = np.zeros(_batch_shape)
        _vol[label_batch == 0] = 1
        _vol = _vol[..., np.newaxis]
        for i in range(self._num_cls):
            if i == 0:
                continue
            _n_slice = np.zeros(label_batch.shape)
            _n_slice[label_batch == i] = 1
            _vol = np.concatenate( (_vol, _n_slice[..., np.newaxis]), axis = 3 )
        return np.float32(_vol)
 
    def vectorCosine(self, x, y):
        """Calculate the cosine similarity between corresponding pixels"""
        tmp1 = np.sum(np.multiply(x,y),axis=3)
        tmp2 = np.sum(np.multiply(x,x),axis=3)
        tmp3 = np.sum(np.multiply(y,y),axis=3)
        return tmp1/np.multiply(np.sqrt(tmp2), np.sqrt(tmp3))
        
        
    def test(self):
        """Test Function."""

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        test_list = self.read_lists(self.test_fid)

        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, self.checkpoint_pth)

            dice_list = []
            assd_list = []
            for idx_file, fid in enumerate(test_list):
                _npz_dict = np.load(fid)
                data = _npz_dict['arr_0']
                label = _npz_dict['arr_1']

                # This is to make the orientation of test data match with the training data
                # Set to False if the orientation of test data has already been aligned with the training data
                if True:
                    data = np.flip(data, axis=0)
                    data = np.flip(data, axis=1)
                    label = np.flip(label, axis=0)
                    label = np.flip(label, axis=1)

                tmp_pred = np.zeros(label.shape)
                frame_list = [kk for kk in range(data.shape[2])]
                
                for ii in range(int(np.floor(data.shape[2] // self.batch_size))):
                    
                    data_batch = np.zeros([self.batch_size, data_size[0], data_size[1], data_size[2]])#256*256*1
                    data_batch_before = np.zeros([self.batch_size, data_size[0], data_size[1], data_size[2]])#256*256*1
                    data_batch_after = np.zeros([self.batch_size, data_size[0], data_size[1], data_size[2]])#256*256*1
                    label_batch = np.zeros([self.batch_size, label_size[0], label_size[1]])#256*256
                    for idx, jj in enumerate(frame_list[ii * self.batch_size: (ii + 1) * self.batch_size]):
                        
                        if idx==0:
                            data_batch_before[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)
                            data_batch[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)
                            data_batch_after[idx, ...] = np.expand_dims(data[..., jj+1].copy(), 2)
                        elif idx==127:
                            data_batch_before[idx, ...] = np.expand_dims(data[..., jj-1].copy(), 2)
                            data_batch[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)
                            data_batch_after[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)
                        else:
                            data_batch_before[idx, ...] = np.expand_dims(data[..., jj-1].copy(), 2)
                            data_batch[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)
                            data_batch_after[idx, ...] = np.expand_dims(data[..., jj+1].copy(), 2)
                        label_batch[idx, ...] = label[..., jj].copy()

                    
                    label_batch = self.label_decomp(label_batch)
                    if TEST_MODALITY=='CT':
                        data_batch_before = np.subtract(np.multiply(np.divide(np.subtract(data_batch_before, -2.8), np.subtract(3.2, -2.8)), 2.0),1)
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -2.8), np.subtract(3.2, -2.8)), 2.0),1)
                        data_batch_after = np.subtract(np.multiply(np.divide(np.subtract(data_batch_after, -2.8), np.subtract(3.2, -2.8)), 2.0),1)
                    elif TEST_MODALITY=='MR':
                        data_batch_before = np.subtract(np.multiply(np.divide(np.subtract(data_batch_before, -1.8), np.subtract(4.4, -1.8)), 2.0),1)
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -1.8), np.subtract(4.4, -1.8)), 2.0),1)
                        data_batch_after = np.subtract(np.multiply(np.divide(np.subtract(data_batch_after, -1.8), np.subtract(4.4, -1.8)), 2.0),1)
                                
                    predicter_b = sess.run(self.predicter_b, feed_dict={self.input_b: data_batch, self.gt_b: label_batch})
                    predicter_b_before = sess.run(self.predicter_b, feed_dict={self.input_b: data_batch_before})
                    predicter_b_after = sess.run(self.predicter_b, feed_dict={self.input_b: data_batch_after})
                    
                    cos1 = np.tile(np.expand_dims(self.vectorCosine(predicter_b, predicter_b_before), 3), (1, 1, 1, 5))
                    cos2 = np.tile(np.expand_dims(self.vectorCosine(predicter_b, predicter_b_after), 3), (1, 1, 1, 5))
                    
                    #Use information of adjacent slices to calibrate the prediction of the current slice 
                    beta = -1
                    p_b = beta*predicter_b + (1-beta)*(cos1/(cos1+cos2)*predicter_b_before + cos2/(cos1+cos2)*predicter_b_after)
                    compact_pred_b_val = (tf.argmax(p_b, 3)).eval()
        
                    for idx, jj in enumerate(frame_list[ii * self.batch_size: (ii + 1) * self.batch_size]):
                        tmp_pred[..., jj] = compact_pred_b_val[idx, ...].copy()
                    
                  
                        
                        
                for c in range(1, self._num_cls):
                    pred_test_data_tr = tmp_pred.copy()
                    pred_test_data_tr[pred_test_data_tr != c] = 0

                    pred_gt_data_tr = label.copy()
                    pred_gt_data_tr[pred_gt_data_tr != c] = 0

                    dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                    try:
                        assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))
                    except:
                        print("error")
                        assd_list.append(100)

            dice_arr = 100 * np.reshape(dice_list, [4, -1]).transpose()

            dice_mean = np.mean(dice_arr, axis=1)
            dice_std = np.std(dice_arr, axis=1)

            print('Dice:')
            print('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
            print('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
            print('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
            print('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
            print('Mean:%.2f' % np.mean(dice_mean))

            assd_arr = np.reshape(assd_list, [4, -1]).transpose()

            assd_mean = np.mean(assd_arr, axis=1)
            assd_std = np.std(assd_arr, axis=1)

            print('ASSD:')
            print('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
            print('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
            print('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
            print('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
            print('Mean:%.2f' % np.mean(assd_mean))

            
def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    adr_model = ADR(config)
    adr_model.test()

if __name__ == '__main__':
    main(config_filename='./config_param.json')
