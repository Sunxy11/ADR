#-*- coding:utf-8 -*-

import numpy as np
import nibabel as nib
import os
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import tucker

import matplotlib.pyplot as plt



import cv2 as cv
#tf.enable_eager_execution()


'''
img = nib.load('./data/test_mr_raw/image_mr_1007.nii.gz')
label = nib.load('./data/test_mr_raw/gth_mr_1007.nii.gz')
print(img.shape)
print(label.shape)
#Convert them to numpy format,
data = img.get_fdata()
label_data = label.get_fdata()

np.savez('./data/test_mr/mr_1007.npz',data,label_data)
'''


tl.set_backend('numpy')
os.environ["CUDA_VISIBLE_DEVICES"] = "9" 


"""Code for training SIFA."""
from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import time

import tensorflow as tf

import data_loader, losses, model
from stats_func import *
import cv2
from skimage import img_as_ubyte

os.environ['CUDA_VISIBLE_DEVICES'] = '9'

save_interval = 300
evaluation_interval = 10
random_seed = 1234


class SIFA:
    """The SIFA module."""

    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._source_train_pth = config['source_train_pth']
        self._target_train_pth = config['target_train_pth']
        self._source_val_pth = config['source_val_pth']
        self._target_val_pth = config['target_val_pth']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._max_step = int(config['max_step'])
        self._keep_rate_value = float(config['keep_rate_value'])
        self._is_training_value = bool(config['is_training_value'])
        self._batch_size = int(config['batch_size'])
     
    
    def train(self):

        # Load Dataset
        self.inputs = data_loader.load_data(self._source_train_pth, self._target_train_pth, True)
        self.inputs_val = data_loader.load_data(self._source_val_pth, self._target_val_pth, True)

        
        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        with open(self._source_train_pth, 'r') as fp:
            rows_s = fp.readlines()
        with open(self._target_train_pth, 'r') as fp:
            rows_t = fp.readlines()

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
            sess.run(init)


            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            curr_lr_seg = 0.001
            cnt = 0
            
            """
            sumsum = [0., 0., 0., 0., 0.]
            for i in range(1200):
                starttime = time.time()

                cnt += 1

                images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                
                for j in range(5):
                    
                    sumsum[j] = tf.add(sumsum[j], tf.reduce_sum(gts_i[:, :, :, j]))
            print('BG', sumsum[0].eval())
            print('MYO', sumsum[1].eval())
            print('LAC', sumsum[2].eval())
            print('LVC', sumsum[3].eval())
            print('AA', sumsum[4].eval())  
            """
            """
            for i in range(1200):
                cnt += 1
                images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                for j in range(self._batch_size):
                    current = np.count_nonzero(gts_i[j,:,:,:] != 0)
                    if current==0:
                        cnt += 1
            print('pos', (1200*self._batch_size-cnt))
            print('neg', cnt)    
            """
                
               
               

                for jj in range(self._batch_size):
                    image = tf.reshape(images_i[jj],[256,256,1])
                    
                    img=image.eval().astype(np.float32)
                    #print(img.min())
                    #print(img.max())
                    img_255=((img-img.min())*(1/img.max()-img.min())).reshape((256,256))
                    img_color = cv2.applyColorMap((img_255*255).astype(np.uint8), cv2.COLORMAP_JET)
                    
                    
                    
                    #image = np.tile(img, [1, 1, 3])
                    #X=np.empty((3, 4, 2))
                    image = np.expand_dims(img,axis=0)
                    image = np.tile(image, [2, 1, 1, 3])
                    #print(image.shape)
                    print('down image eval')

                    
                    
'''
                    h = 256
                    w = 256
                    #src = cv.GaussianBlur(img, (0, 0), 1)
                    dst = cv.Laplacian(img_color, cv.CV_32F, ksize=3, delta=127)
                    dst = cv.convertScaleAbs(dst)
                    result = np.zeros([h, w*2, 1], dtype=img.dtype)
                    #result[0:h,0:w,:] = image
                    #result[0:h,w:2*w,:] = dst
                    plt.subplot(121)
                    plt.imshow(img, cmap='gray')
                    plt.subplot(122)
                    plt.imshow(dst, cmap='gray')
                    plt.show()
'''
                    
                    

                    core, factors = tucker(image.astype(np.float32),rank=[1,60,64,2])
                    print(core.shape)
                    b0=tl.tenalg.mode_dot(core,factors[0],0)
                    print(factors[0].shape)
                    print(b0.shape)
                    print('*******************************')
                    b1=tl.tenalg.mode_dot(core,factors[1],1)
                    print(factors[1].shape)
                    print(b1.shape)
                    print('*******************************')
                    b2=tl.tenalg.mode_dot(core,factors[2],2)
                    print(factors[2].shape)
                    print(b2.shape)
                    print('*******************************')
                    b3=tl.tenalg.mode_dot(core,factors[3],3)
                    print(factors[3].shape)
                    print('ddddddddddddd:',b3.shape)
                    print('*******************************')
                    fac=[]
                    fac.append(factors[0])
                    fac.append(factors[1])
                    fac.append(factors[2])
                    print(tl.tenalg.multi_mode_dot(core,fac).shape)

                    
                    
                    
                    
                    
                    
'''
                    fig = plt.figure(figsize=(8, 4))       
                    ax1 = fig.add_subplot(331)
                    ax1.set_title('image')
                    #ss=image[:,:,0].reshape((256,256))
                    ax1.imshow(img.reshape((256,256)), cmap='gray')
                    
                    ax2 = fig.add_subplot(332)
                    ax2.set_title('1')
                    #ax2.imshow(core[:,:,0], cmap='gray')
                    ax2.imshow(img_255, cmap='gray')
                    
                    ax3 = fig.add_subplot(333)
                    ax3.set_title('2')
                    ax3.imshow(img_color)
                    
                    ax4 = fig.add_subplot(334)
                    #ax4.set_title('factor1')
                    ax4.imshow(tl.tenalg.mode_dot(core,factors[0],0))
                    
                    ax5 = fig.add_subplot(335)
                    #ax5.set_title('factor2')
                    ax5.imshow(tl.tenalg.mode_dot(core,factors[1],1))
                    
                    ax6 = fig.add_subplot(336)
                    #ax6.set_title('factor3')
                    ax6.imshow(tl.tenalg.mode_dot(core,factors[2],2))
                    
                    ax7 = fig.add_subplot(337)
                    #ax7.set_title('core1')
                    ax7.imshow(tl.tenalg.multi_mode_dot(core,factors))
                    
                    ax8 = fig.add_subplot(338)
                    #ax8.set_title('core2*1')
                    #cdf=tl.tenalg.mode_dot(core,factors[0],0)
                    #print(cdf.shape)
                    fac=[]
                    fac.append(factors[0])
                    fac.append(factors[1)
                    ax8.imshow(tl.tenalg.multi_mode_dot(core,fac))
                    
                    ax9 = fig.add_subplot(339)
                    ax9.set_title('label')
                   
                    gts=tf.reshape(gts_i[jj],[256,256,5])
                    gts=tf.argmax(gts,2)
                    gts=gts.eval().astype(np.float32)
                    ax9.imshow(gts)
                    
                    fig.savefig('./tucker_pictures/tucker_core_'+str(i)+'_'+str(jj)+'.png')
'''
            
           
            


def main(config_filename):
    
    tf.set_random_seed(random_seed)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    sifa_model = SIFA(config)
    sifa_model.train()


if __name__ == '__main__':
    main(config_filename='./config_param.json')



















'''
source_pth='./data/datalist/training_ct.txt'
decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 3]
        'label_vol': tf.FixedLenFeature([], tf.string)}
imagea_list=[]
raw_size = [256, 256, 3]
volume_size = [256, 256, 3]
label_size = [256, 256, 1]
reader = tf.TFRecordReader()
with tf.Session() as sess:
    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
        for row in rows:
            
            path=[]
            index = row.find('tfrecords')
            #try:
            path.append(row[:index+9])
            print(path)
            data_queue = tf.train.string_input_producer(path)
            _,serialized_example = reader.read(data_queue)
               
            parser = tf.parse_single_example(serialized_example,features=decomp_feature)   
            
            data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
            data_vol = tf.reshape(data_vol, raw_size)
            data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)
            label_vol = tf.decode_raw(parser['label_vol'], tf.float32) 
            label_vol = tf.reshape(label_vol, raw_size)
            label_vol = tf.slice(label_vol, [0, 0, 1], label_size)
            batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)
            image=tf.expand_dims(data_vol[:, :, 1], axis=2)
            label=batch_y
            print(sess.run(image))
            print('3*********************************************************************************')
            image = tf.reshape(image,[256,256,1])
            print(image)
            print('1*********************************************************************************') 
            print(image.eval())
            print('2*********************************************************************************')
            print('down image eval')
            img = np.tile(image, [1, 1, 3])
            core, factors = tucker(img,rank=[64,64,2])
            print(core)
            
            fig = plt.figure(figsize=(8, 4))       
            ax1 = fig.add_subplot(121)
            ax1.set_title('image')
            ss=image.reshape((256,256))
            ax1.imshow(ss, cmap='gray')
            
            ax2 = fig.add_subplot(122)
            ax2.set_title('core')
            ax2.imshow(core, cmap='gray')
            
            fig.savefig('./tucker_core.png')
            
          
            #print(image.shape,label.shape,image.dtype,label.dtype)
            #print('YES')


'''



    
'''        
def _decode_samples(image_list, shuffle=False):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 3]
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1] # the label has size [256,256,3] in the preprocessed data, but only the middle slice is used

    parser = tf.parse_single_example(image_list, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)

    batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)
    return tf.expand_dims(data_vol[:, :, 1], axis=2), batch_y
            
        
with tf.Session() as sess:
    num_epochs=1
    training_filenames = ["./data/mr_train_tfs/mr_train_slice999.tfrecords"]
    dataset = tf.data.TFRecordDataset(training_filenames)
    dataset = dataset.map(_decode_samples)
    dataset = dataset.shuffle(buffer_size=10000) 
    dataset = dataset.batch(1)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    image, labels = iterator.get_next()

    image=tf.convert_to_tensor(image)
    X = tl.tensor(np.arange(24).reshape(3, 4, 2))
    
    image = tf.reshape(image,[256,256,1])
    image = image.eval()
    
    
    core, factors = tucker(image,rank=[3,4,2])
    
    
    print(core)
  
    
    data_provider = Provider()
    data_provider.full_tensor = lambda: image
    env = Environment(data_provider, summary_path='/tmp/cp_demo_' + '30')
    cp = CP_ALS(env)
    # set rank=10 for decomposition
    args = CP_ALS.CP_Args(rank=10, validation_internal=1)
    # build decomposition model with arguments
    cp.build_model(args)
    # train decomposition model, set the max iteration as 100
    cp.train(100)
    # obtain factor matrices from trained model
    factor_matrices = cp.factors
    for matrix in factor_matrices:
        print(matrix)
    # obtain scaling vector from trained model
    lambdas = cp.lambdas
    print(lambdas)
'''
'''
    fig = plt.figure(figsize=(8, 4))       
    ax1 = fig.add_subplot(121)
    ax1.set_title('image')
    ss=image.reshape((256,256))
    ax1.imshow(image, cmap='gray')
        
    ax2 = fig.add_subplot(122)
    ax2.set_title('core')
    ax2.imshow(core[0], cmap='gray')
    
    fig.savefig('./tucker_core.png')
'''
  


'''
# use synthetic_data_cp to generate a random tensor with shape of 40x40x40
X = dg.synthetic_data_cp([40, 40, 40], 10)
data_provider = Provider()
data_provider.full_tensor = lambda: X
env = Environment(data_provider, summary_path='/tmp/cp_demo_' + '30')
cp = CP_ALS(env)
# set rank=10 for decomposition
args = CP_ALS.CP_Args(rank=10, validation_internal=1)
# build decomposition model with arguments
cp.build_model(args)
# train decomposition model, set the max iteration as 100
cp.train(100)
# obtain factor matrices from trained model
factor_matrices = cp.factors
for matrix in factor_matrices:
    print(matrix)
# obtain scaling vector from trained model
lambdas = cp.lambdas
print(lambdas)
'''


