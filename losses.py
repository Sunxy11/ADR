import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from typing import Tuple, Iterable, Set, cast



def lsgan_loss_generator(prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the generator.
    """
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_feak_b, prob_real_b):
    """
    Computes the LS-GAN loss as minimized by the discriminator.
    """
    return (tf.reduce_mean(tf.squared_difference(prob_feak_b, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_real_b, 0))) * 0.5
      

    
def HSIC_lossfunc(x, y, b_s):
    assert x.shape.ndims == y.shape.ndims == 2
    m = b_s
    h = tf.eye(m) - 1/m
    K_x = gaussian_kernel(x)
    K_y = gaussian_kernel(y)
    return tf.trace(tf.matmul(tf.matmul(tf.matmul(K_x, h), K_y), h)) / (m-1+1e-10)

def gaussian_kernel(x, y=None, sigma=5):
    if y is None:
        y = x
    assert x.shape.ndims == y.shape.ndims == 2
    assert x.shape == y.shape
    z = tf.reduce_sum(((tf.expand_dims(x, 0) - tf.expand_dims(x, 1)) ** 2),-1)
    return tf.exp(- 0.5 * z / (sigma * sigma))




def _softmax_weighted_loss(logits, gt):
    """
    Calculate weighted cross-entropy loss.
    """
    softmaxpred = tf.nn.softmax(logits)
    for i in range(5):
        gti = gt[:,:,:,i]
        predi = softmaxpred[:,:,:,i]
        weighted = 1-(tf.reduce_sum(gti) / tf.reduce_sum(gt))
        if i == 0:
            raw_loss = -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))

    loss = tf.reduce_mean(raw_loss)

    return loss


def _dice_loss_fun(logits, gt):
    """
    Calculate dice loss.
    """
    dice = 0
    eps = 1e-7
    softmaxpred = tf.nn.softmax(logits)
    for i in range(5):
        inse = tf.reduce_sum(softmaxpred[:, :, :, i]*gt[:, :, :, i])
        l = tf.reduce_sum(softmaxpred[:, :, :, i]*softmaxpred[:, :, :, i])
        r = tf.reduce_sum(gt[:, :, :, i])
        dice += 2.0 * inse/(l+r+eps)

    return 1 - 1.0 * dice / 5
    
    

def task_loss(prediction, gt, batch_size=None):
    """
    Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    ce_loss = _softmax_weighted_loss(prediction, gt)
    dice_loss = _dice_loss_fun(prediction, gt)
    
    return ce_loss, dice_loss


