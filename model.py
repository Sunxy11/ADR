
import numpy as np
import tensorflow as tf
import layers
import json
import random
import matplotlib.pyplot as plt
import cv2

with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
POOL_SIZE = int(config['pool_size'])

# The height of each image.
IMG_HEIGHT = 256
# The width of each image.
IMG_WIDTH = 256

ngf = 32
ndf = 64



def get_outputs(inputs, bs, skip=False, is_training=True, keep_rate=0.75):
    images_a = inputs['images_a']
    images_b = inputs['images_b']

    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:
        
        content_encoder = ContentEncoder
        current_segmenter = build_segmenter
        current_spf_classifier=build_spf_cls
        current_discriminator = discriminator
        
        fake_images_b = build_generator_resnet_9blocks(images_a, images_a, name='g_A', skip=skip)
        
        dis_real_b = discriminator_b(images_b, "d_B")
        dis_fake_b = discriminator_b(fake_images_b, "d_B")
        
        inv_a, spf_a, logist_inv_a, logist_spf_a, inv_a_aux, spf_a_aux, logist_inv_a_aux, logist_spf_a_aux, Attn_a, Attn_a_aux = content_encoder(fake_images_b, bs, name='z', skip=skip, is_training=is_training, keep_rate=keep_rate)
        inv_b, spf_b, logist_inv_b, logist_spf_b, inv_b_aux, spf_b_aux, logist_inv_b_aux, logist_spf_b_aux, Attn_b, Attn_b_aux = content_encoder(images_b, bs, name='z', skip=skip, is_training=is_training, keep_rate=keep_rate)
        
        pred_real_a = current_segmenter(inv_a, 'seg', keep_rate=keep_rate)
        pred_real_b = current_segmenter(inv_b, 'seg', keep_rate=keep_rate)
        
        pred_real_a_aux = current_segmenter(inv_a_aux, 'seg_aux', keep_rate=keep_rate)
        pred_real_b_aux = current_segmenter(inv_b_aux, 'seg_aux', keep_rate=keep_rate)

        dis_a = tf.multiply(tf.nn.softmax(pred_real_a), Attn_a)
        dis_b = tf.multiply(tf.nn.softmax(pred_real_b), Attn_b)
        dis_pred_real_a = current_discriminator(dis_a, "d")
        dis_pred_real_b = current_discriminator(dis_b, 'd')
       
        dis_a_aux = tf.multiply(tf.nn.softmax(pred_real_a_aux), Attn_a_aux)
        dis_b_aux = tf.multiply(tf.nn.softmax(pred_real_b_aux), Attn_b_aux)
        dis_pred_real_a_aux = current_discriminator(dis_a_aux, "d_aux")
        dis_pred_real_b_aux = current_discriminator(dis_b_aux, 'd_aux')
        
        cls_spf_a = current_spf_classifier(spf_a, 'cls_s', is_training=is_training, keep_rate=keep_rate)
        cls_spf_b = current_spf_classifier(spf_b, 'cls_s', is_training=is_training, keep_rate=keep_rate)
        
    return {
        'spf_a': spf_a,
        'logist_spf_a': logist_spf_a,
        'spf_b': spf_b,
        'logist_spf_b':logist_spf_b,
        'inv_a': inv_a,
        'logist_inv_a':logist_inv_a,
        'inv_b': inv_b,
        'logist_inv_b':logist_inv_b,
        'pred_real_a': pred_real_a,
        'pred_real_b': pred_real_b,
        'dis_pred_real_a': dis_pred_real_a,
        'dis_pred_real_b': dis_pred_real_b,
        
        'spf_a_aux': spf_a_aux,
        'logist_spf_a_aux': logist_spf_a_aux,
        'spf_b_aux': spf_b_aux,
        'logist_spf_b_aux':logist_spf_b_aux,
        'inv_a_aux': inv_a_aux,
        'logist_inv_a_aux':logist_inv_a_aux,
        'inv_b_aux': inv_b_aux,
        'logist_inv_b_aux':logist_inv_b_aux,
        'pred_real_a_aux': pred_real_a_aux,
        'pred_real_b_aux': pred_real_b_aux,
        'dis_pred_real_a_aux': dis_pred_real_a_aux,
        'dis_pred_real_b_aux': dis_pred_real_b_aux,
        
        'fake_images_b': fake_images_b,
        'dis_real_b': dis_real_b,
        'dis_fake_b': dis_fake_b,
        
        'cls_spf_a': cls_spf_a,
        'cls_spf_b': cls_spf_b,
    }

def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim, 3, 3, 1, 1, 0.01, "VALID", "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim, 3, 3, 1, 1, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)

        return tf.nn.relu(out_res + inputres)


def build_resnet_block_ins(inputres, dim, name="resnet", padding="REFLECT"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d_ga(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", norm_type='Ins')
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d_ga(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False, norm_type='Ins')

        return tf.nn.relu(out_res + inputres)


def build_resnet_block_ds(inputres, dim_in, dim_out, name="resnet", padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim_out, 3, 3, 1, 1, 0.01, "VALID", "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim_out, 3, 3, 1, 1, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        inputres = tf.pad(inputres, [[0, 0], [0, 0], [0, 0], [(dim_out - dim_in) // 2, (dim_out - dim_in) // 2]], padding)

        return tf.nn.relu(out_res + inputres)


def build_drn_block(inputdrn, dim, name="drn", padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_drn = tf.pad(inputdrn, [[0, 0], [2, 2], [2, 2], [0, 0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim, dim, 3, 3, 2, 0.01, "VALID", "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_drn = tf.pad(out_drn, [[0, 0], [2, 2], [2, 2], [0, 0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim, dim, 3, 3, 2, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)

        return tf.nn.relu(out_drn + inputdrn)


def build_drn_block_ds(inputdrn, dim_in, dim_out, name='drn_ds', padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_drn = tf.pad(inputdrn, [[0,0], [2,2], [2,2], [0,0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim_in, dim_out, 3, 3, 2, 0.01, 'VALID', "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_drn = tf.pad(out_drn, [[0,0], [2,2], [2,2], [0,0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim_out, dim_out, 3, 3, 2, 0.01, 'VALID', "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        inputdrn = tf.pad(inputdrn, [[0,0], [0,0], [0, 0], [(dim_out-dim_in)//2,(dim_out-dim_in)//2]], padding)

        return tf.nn.relu(out_drn + inputdrn)


def discriminator_b(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        padw = 2
        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
     
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2, 0.02, "VALID", "c1", do_norm=False, relufactor=0.2, norm_type='Ins')
   
        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2, 0.02, "VALID", "c2", relufactor=0.2, norm_type='Ins')
    
        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2, 0.02, "VALID", "c3", relufactor=0.2, norm_type='Ins')
    
        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1, 0.02, "VALID", "c4", relufactor=0.2, norm_type='Ins')

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(pad_o_c4, 1, f, f, 1, 1, 0.02, "VALID", "c5", do_norm=False, do_relu=False)

        return o_c5


def discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        padw = 2
   
        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
     
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2, 0.02, "VALID", "c1", do_norm=False, relufactor=0.2, norm_type='Batch')
   
        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2, 0.02, "VALID", "c2", relufactor=0.2, norm_type='Batch')
    
        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2, 0.02, "VALID", "c3", relufactor=0.2, norm_type='Batch')
    
        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1, 0.02, "VALID", "c4", relufactor=0.2, norm_type='Batch')

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(pad_o_c4, 1, f, f, 1, 1, 0.02, "VALID", "c5", do_norm=False, do_relu=False)
 
        return o_c5


def build_segmenter(inputse, name='segmenter', keep_rate=0.75):
    with tf.variable_scope(name):
        k1 = 1

        o_c8 = layers.general_conv2d(inputse, 5, k1, k1, 1, 1, 0.01, 'SAME', 'c8', do_norm=False, do_relu=False, keep_rate=keep_rate)
        out_seg = tf.image.resize_images(o_c8, (256, 256))
        
        return out_seg


def ContentEncoder(inputen, bs, name='ContentEncoder', skip=False, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        fb = 16
        k1 = 3
        padding = "CONSTANT"

        o_c1 = layers.general_conv2d(inputen, fb, 7, 7, 1, 1, 0.01, 'SAME', name="c1", norm_type="Batch", is_training=is_training, keep_rate=keep_rate)
        o_r1 = build_resnet_block(o_c1, fb, "r1", padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        out1 = tf.nn.max_pool(o_r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r2 = build_resnet_block_ds(out1, fb, fb*2, "r2", padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        out2 = tf.nn.max_pool(o_r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r3 = build_resnet_block_ds(out2, fb*2, fb*4, 'r3', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r4 = build_resnet_block(o_r3, fb*4, 'r4', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        out3 = tf.nn.max_pool(o_r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r5 = build_resnet_block_ds(out3, fb*4, fb*8, 'r5', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r6 = build_resnet_block(o_r5, fb*8, 'r6', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_r7 = build_resnet_block_ds(o_r6, fb*8, fb*16, 'r7', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r8 = build_resnet_block(o_r7, fb*16, 'r8', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_r9 = build_resnet_block(o_r8, fb*16, 'r9', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r10 = build_resnet_block(o_r9, fb * 16, 'r10', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_r11 = build_resnet_block_ds(o_r10, fb * 16, fb * 32, 'r11', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r12 = build_resnet_block(o_r11, fb * 32, 'r12', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_d1 = build_drn_block(o_r12, fb*32, 'd1', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_d2 = build_drn_block(o_d1, fb*32, 'd2', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_c2 = layers.general_conv2d(o_d2, fb*32, k1, k1, 1, 1, 0.01, 'SAME', 'c2', norm_type='Batch', is_training=is_training,keep_rate=keep_rate)
        o_c3 = layers.general_conv2d(o_c2, fb*32, k1, k1, 1, 1, 0.01, 'SAME', 'c3', norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        
        z_inv=o_c3[...,:256]
        z_spf=o_c3[...,256:]
        logist_inv = tf.nn.avg_pool2d(z_inv,ksize=[1,z_inv.get_shape().as_list()[1],z_inv.get_shape().as_list()[1],1],strides=[1,1,1,1],padding='VALID',name='GAP')
        logist_spf = tf.nn.avg_pool2d(z_spf,ksize=[1,z_spf.get_shape().as_list()[1],z_spf.get_shape().as_list()[1],1],strides=[1,1,1,1],padding='VALID',name='GAP')
        
        z_inv_aux=o_r12[...,:256]
        z_spf_aux=o_r12[...,256:]
        logist_inv_aux = tf.nn.avg_pool2d(z_inv_aux,ksize=[1,z_inv_aux.get_shape().as_list()[1],z_inv_aux.get_shape().as_list()[1],1],strides=[1,1,1,1],padding='VALID',name='GAP')
        logist_spf_aux = tf.nn.avg_pool2d(z_spf_aux,ksize=[1,z_spf_aux.get_shape().as_list()[1],z_spf_aux.get_shape().as_list()[1],1],strides=[1,1,1,1],padding='VALID',name='GAP')
        
        Attn = tf.reshape(tf.reduce_max(z_inv, axis=3),[bs,32,32,1])
        Attn = tf.image.resize_images(Attn, (256, 256))
        Attn = (v - tf.reduce_min(Attn)) / (tf.reduce_max(Attn) - tf.reduce_min(Attn))# * 255
        Attn = tf.tile(Attn, [1,1,1,5])
        Attn_aux = tf.reshape(tf.reduce_max(z_inv_aux, axis=3),[bs,32,32,1])
        Attn_aux = tf.image.resize_images(Attn_aux, (256, 256))
        Attn_aux = (Attn_aux - tf.reduce_min(Attn_aux)) / (tf.reduce_max(Attn_aux) - tf.reduce_min(Attn_aux))# *255
        Attn_aux = tf.tile(Attn_aux, [1,1,1,5])
 
        
        return z_inv,z_spf,tf.reshape(logist_inv,[bs,256]),tf.reshape(logist_spf,[bs,256]),z_inv_aux,z_spf_aux,tf.reshape(logist_inv_aux,[bs,256]),tf.reshape(logist_spf_aux,[bs,256]),Attn,Attn_aux
        
        
def build_spf_cls(inputse, name='spf_classifier', is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        ks = 3
        pad_total = ks - 1
        pad_begin = pad_total // 2
        pad_end = pad_total - pad_begin
        
        num_classes = 2
        
        pad_input = tf.pad(inputse, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]], "CONSTANT")
        
        o2 = layers.general_conv2d(pad_input, 512, ks, ks, 2, 2, 0.01, 'VALID', 'o2', norm_type='Batch', is_training=is_training,keep_rate=keep_rate)
        o3 = tf.nn.avg_pool2d(o2,ksize=[1,o2.get_shape().as_list()[1],o2.get_shape().as_list()[1],1],strides=[1,1,1,1],padding='VALID',name='GAP')
        flatten_c = tf.layers.flatten(o3)
        cls = tf.layers.dense(flatten_c, 128, activation='relu')
        cls = tf.layers.dense(cls, num_classes, activation=None)
        
        return cls
      
def build_generator_resnet_9blocks(inputgen, inputimg, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"
        
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d_ga(pad_input, ngf, f, f, 1, 1, 0.02, name="c1", norm_type='Ins')
        o_c2 = layers.general_conv2d_ga(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", norm_type='Ins')
        o_c3 = layers.general_conv2d_ga(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", norm_type='Ins')

        o_r1 = build_resnet_block_ins(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block_ins(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block_ins(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block_ins(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block_ins(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block_ins(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block_ins(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block_ins(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block_ins(o_r8, ngf * 4, "r9", padding)
       
        o_c4 = layers.general_deconv2d(o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4", norm_type='Ins')
        o_c5 = layers.general_deconv2d(o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5", norm_type='Ins')
        o_c6 = layers.general_conv2d_ga(o_c5, 1, f, f, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False)
       
        if skip is True:
            out_gen = tf.nn.tanh(inputimg + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen
        
    
