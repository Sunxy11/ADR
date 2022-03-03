"""Code for training ADR."""
from datetime import datetime
import json
import numpy as np
import random
import os
import time

import tensorflow as tf

import data_loader, losses
import model as model_output
from stats_func import *


os.environ['CUDA_VISIBLE_DEVICES'] = '9'

save_interval = 100
evaluation_interval = 10
random_seed = 1234




class ADR:
    

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
        self._output_select_dir = os.path.join('./output_select', current_time)
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._max_step = int(config['max_step'])
        self._keep_rate_value = float(config['keep_rate_value'])
        self._is_training_value = bool(config['is_training_value'])
        self._batch_size = int(config['batch_size'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']
        self._weight_decay = float(config['weight_decay'])

    def model_setup(self):

        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                1
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                1
            ], name="input_B")
       
        
        self.gt_a = tf.placeholder(
            tf.float32, [
                None,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                self._num_cls
            ], name="gt_A")
        self.gt_b = tf.placeholder(
            tf.float32, [
                None,
                model_output.IMG_WIDTH,
                model_output.IMG_HEIGHT,
                self._num_cls
            ], name="gt_B") 
        
        self.cls_label_a = tf.placeholder(
            tf.float32, [
                self._batch_size,
                2
            ], name="cls_label_A")
        self.cls_label_b = tf.placeholder(
            tf.float32, [
                self._batch_size,
                2
            ], name="cls_label_B")
        
            
        
        self.keep_rate = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.num_fake_inputs = 0

        self.learning_rate_dis = tf.placeholder(tf.float32, shape=[], name="lr_dis")
        self.learning_rate_seg = tf.placeholder(tf.float32, shape=[], name="lr_seg")
        self.learning_rate_d = tf.placeholder(tf.float32, shape=[], name="lr_d")
        self.learning_rate_cls = tf.placeholder(tf.float32, shape=[], name="lr_cls")
        self.learning_rate_gd = tf.placeholder(tf.float32, shape=[], name="lr_gd")

     
        self.lr_gd_summ = tf.summary.scalar("lr_gd", self.learning_rate_gd)

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
        }

        outputs = model_output.get_outputs(inputs, self._batch_size, skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)

        self.spf_a = outputs['spf_a']
        self.spf_b = outputs['spf_b']
        self.inv_a = outputs['inv_a']
        self.inv_b = outputs['inv_b']
        self.logist_spf_a = outputs['logist_spf_a']
        self.logist_spf_b = outputs['logist_spf_b']
        self.logist_inv_a = outputs['logist_inv_a']
        self.logist_inv_b = outputs['logist_inv_b']

        self.dis_pred_real_a = outputs['dis_pred_real_a']
        self.dis_pred_real_b = outputs['dis_pred_real_b']
        
        self.pred_real_a = outputs['pred_real_a']
        self.pred_real_b = outputs['pred_real_b']
        
        self.spf_a_aux = outputs['spf_a_aux']
        self.spf_b_aux = outputs['spf_b_aux']
        self.inv_a_aux = outputs['inv_a_aux']
        self.inv_b_aux = outputs['inv_b_aux']
        self.logist_spf_a_aux = outputs['logist_spf_a_aux']
        self.logist_spf_b_aux = outputs['logist_spf_b_aux']
        self.logist_inv_a_aux = outputs['logist_inv_a_aux']
        self.logist_inv_b_aux = outputs['logist_inv_b_aux']

        self.dis_pred_real_a_aux = outputs['dis_pred_real_a_aux']
        self.dis_pred_real_b_aux = outputs['dis_pred_real_b_aux']
        
        self.pred_real_a_aux = outputs['pred_real_a_aux']
        self.pred_real_b_aux = outputs['pred_real_b_aux']
       
       
        self.fake_images_b = outputs['fake_images_b']
        self.dis_real_b = outputs['dis_real_b']
        self.dis_fake_b = outputs['dis_fake_b']
        
      
        
        self.cls_spf_a = outputs['cls_spf_a']
        self.cls_spf_b = outputs['cls_spf_b']
        
        
        
        self.predicter_real_a = tf.nn.softmax(self.pred_real_a)
        self.compact_pred_real_a = tf.argmax(self.predicter_real_a, 3)
        self.compact_y_real_a = tf.argmax(self.gt_a, 3)
        
        self.predicter_real_b = tf.nn.softmax(self.pred_real_b)
        self.compact_pred_real_b = tf.argmax(self.predicter_real_b, 3)
        self.compact_y_real_b = tf.one_hot(self.compact_pred_real_b, depth=self._num_cls, axis=-1)  
        
        self.dice_a_arr = dice_eval(self.compact_pred_real_a, self.gt_a, self._num_cls)
        self.dice_a_mean = tf.reduce_mean(self.dice_a_arr)
        self.dice_a_mean_summ = tf.summary.scalar("dice_a", self.dice_a_mean)

        self.dice_b_arr = dice_eval(self.compact_pred_real_b, self.gt_b, self._num_cls)
        self.dice_b_mean = tf.reduce_mean(self.dice_b_arr)
        self.dice_b_mean_summ = tf.summary.scalar("dice_b", self.dice_b_mean)
        
        self.a_val_dice=self.dice_a_mean
        self.b_val_dice=self.dice_b_mean
        
        

    def compute_losses(self):

        
        hsic_loss_a = losses.HSIC_lossfunc(self.logist_spf_a, self.logist_inv_a, self._batch_size)
        hsic_loss_b = losses.HSIC_lossfunc(self.logist_spf_b, self.logist_inv_b, self._batch_size)
        hsic_loss = 0.5 * (hsic_loss_a + hsic_loss_b)
        
        
        spf_domain_loss_a = tf.losses.sigmoid_cross_entropy(logits=self.cls_spf_a, multi_class_labels=self.cls_label_a)
        spf_domain_loss_b = tf.losses.sigmoid_cross_entropy(logits=self.cls_spf_b, multi_class_labels=self.cls_label_b)
        spf_domain_loss = 0.5 * (spf_domain_loss_a + spf_domain_loss_b)

        #task
        ce_loss_real_a, dice_loss_real_a = losses.task_loss(self.pred_real_a, self.gt_a)
        ce_loss_real_a_aux, dice_loss_real_a_aux = losses.task_loss(self.pred_real_a_aux, self.gt_a)
        
    
        
        
        #adv loss
        lsgan_loss_p = losses.lsgan_loss_generator(self.dis_pred_real_b)
        lsgan_loss_p_aux = losses.lsgan_loss_generator(self.dis_pred_real_b_aux)
        
        g_a_loss = losses.lsgan_loss_generator(self.dis_fake_b)
                
        d_loss = losses.lsgan_loss_discriminator(
            prob_feak_b=self.dis_pred_real_a,
            prob_real_b=self.dis_pred_real_b,          
        )
        d_aux_loss = losses.lsgan_loss_discriminator(
            prob_feak_b=self.dis_pred_real_a_aux,
            prob_real_b=self.dis_pred_real_b_aux,      
        )
        
        d_b_loss = losses.lsgan_loss_discriminator(
            prob_feak_b=self.dis_real_b,
            prob_real_b=self.dis_fake_b,      
        )
        
           
        l2_loss_z = self._weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/z/' in v.name])
        l2_loss_seg = self._weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/seg/' in v.name or '/seg_aux/' in v.name])
       
        
        seg_loss =  ce_loss_real_a + dice_loss_real_a + 0.1 * (ce_loss_real_a_aux + dice_loss_real_a_aux) + 0.1 * lsgan_loss_p + 0.01 * lsgan_loss_p_aux + l2_loss_seg
    
        z_loss = 0.1 * hsic_loss + seg_loss + 0.05 * spf_domain_loss + l2_loss_z 
        

        
        optimizer_z = tf.train.AdamOptimizer(self.learning_rate_dis, beta1=0.5)
        optimizer_seg = tf.train.AdamOptimizer(self.learning_rate_seg)
        optimizer_d = tf.train.AdamOptimizer(self.learning_rate_d, beta1=0.5)
        optimizer_cls = tf.train.AdamOptimizer(self.learning_rate_cls, beta1=0.5)
        optimizer_gd = tf.train.AdamOptimizer(self.learning_rate_gd)
        
        self.model_vars = tf.trainable_variables()


        cls_s_vars = [var for var in self.model_vars if '/cls_s/' in var.name]
        seg_vars = [var for var in self.model_vars if '/seg/' in var.name]
        z_vars = [var for var in self.model_vars if '/z/' in var.name]
        d_vars = [var for var in self.model_vars if '/d/' in var.name]
        seg_aux_vars = [var for var in self.model_vars if '/seg_aux/' in var.name]
        d_aux_vars = [var for var in self.model_vars if '/d_aux/' in var.name]
        g_a_vars = [var for var in self.model_vars if '/g_A/' in var.name]
        d_b_vars = [var for var in self.model_vars if '/d_B/' in var.name]

        self.z_trainer = optimizer_z.minimize(z_loss, var_list=z_vars)
        self.seg_trainer = optimizer_seg.minimize(seg_loss, var_list=seg_vars+seg_aux_vars)
        self.d_trainer = optimizer_d.minimize(d_loss, var_list=d_vars)
        self.d_aux_trainer = optimizer_d.minimize(d_aux_loss, var_list=d_aux_vars)
        self.cls_spf_trainer = optimizer_cls.minimize(spf_domain_loss, var_list=cls_s_vars)
        self.g_a_trainer = optimizer_gd.minimize(g_a_loss, var_list=g_a_vars)
        self.d_b_trainer = optimizer_gd.minimize(d_b_loss, var_list=d_b_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.style_content_loss_summ = tf.summary.scalar("style_content_loss", hsic_loss)
        self.dice_A_loss_summ = tf.summary.scalar("dice_A_loss", dice_loss_real_a)
        self.seg_loss_summ = tf.summary.scalar("seg_loss", seg_loss)
        self.z_loss_summ = tf.summary.scalar("z_loss", z_loss)
        self.d_loss_summ = tf.summary.scalar("d_loss", d_loss)
        self.d_aux_loss_summ = tf.summary.scalar("d_aux_loss", d_aux_loss)
        self.lsgan_p_loss_summ = tf.summary.scalar("lsgan_p_loss", lsgan_loss_p)
        self.lsgan_p_aux_loss_summ = tf.summary.scalar("lsgan_p_aux_loss", lsgan_loss_p_aux)
        self.style_domain_loss_summ = tf.summary.scalar("style_domain_loss", spf_domain_loss)
        self.g_a_loss_summ = tf.summary.scalar("g_a_loss", g_a_loss)
        self.d_b_loss_summ = tf.summary.scalar("d_b_loss", d_b_loss)
        
        
    def train(self):

        # Load Dataset
        self.inputs = data_loader.load_data(self._source_train_pth, self._target_train_pth, True)
        self.inputs_val = data_loader.load_data(self._source_val_pth, self._target_val_pth, True)


        global_ = tf.Variable(tf.constant(0),trainable=False)
        lr_gd = tf.train.exponential_decay(self._base_lr, global_,self._max_step, decay_rate=0.9)
        
        
        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()
        
        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5000)

        with open(self._source_train_pth, 'r') as fp:
            rows_s = fp.readlines()
        with open(self._target_train_pth, 'r') as fp:
            rows_t = fp.readlines()

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
            sess.run(init)
            
            # Restore the model to run the model from last checkpoint
            
            if self._to_restore:
                variables = tf.contrib.framework.get_variables_to_restore()
                variables_to_resotre = [v for v in tf.trainable_variables() if '/g_A/' in v.name or '/d_B/' in v.name]
                saver_gd = tf.train.Saver(variables_to_resotre)
                saver_gd.restore(sess, self._checkpoint_dir)
            
           

            writer = tf.summary.FileWriter(self._output_dir)
            writer_val = tf.summary.FileWriter(self._output_dir+'/val')

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            
            cnt = -1

            for i in range(self._max_step):
                starttime = time.time()

                cnt += 1
                curr_lr = self._base_lr
                
                
                images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                
                
                cls_labels_i = np.array([[1., 0.]]).repeat([self._batch_size],axis=0)
                cls_labels_j = np.array([[0., 1.]]).repeat([self._batch_size],axis=0)
                
                
                
                
                inputs = {
                    'images_i': images_i,
                    'images_j': images_j,
                    'gts_i': gts_i,
                    'gts_j': gts_j,
                }    
                    
                    
                images_i_val, images_j_val, gts_i_val, gts_j_val = sess.run(self.inputs_val)
                
                inputs_val = {
                    'images_i_val': images_i_val,
                    'images_j_val': images_j_val,
                    'gts_i_val': gts_i_val,
                    'gts_j_val': gts_j_val,
                }
                
                #trian g_a
                _, summary_str = sess.run(
                    [self.g_a_trainer,
                     self.g_a_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.gt_a:
                            inputs['gts_i'],
                        self.cls_label_a:
                            cls_labels_i,     
                        self.cls_label_b:
                            cls_labels_j,
                        self.learning_rate_gd: 1e-10,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)
                
              
                
                
                #trian d_b
                _, summary_str = sess.run(
                    [self.d_b_trainer,
                     self.d_b_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.gt_a:
                            inputs['gts_i'],
                        self.cls_label_a:
                            cls_labels_i,     
                        self.cls_label_b:
                            cls_labels_j,
                        self.learning_rate_gd: 1e-10,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)
                
                
                #trian z
                _, summary_str, s_c_summ = sess.run(
                    [self.z_trainer,
                     self.z_loss_summ,
                     self.style_content_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.gt_a:
                            inputs['gts_i'],
                        self.cls_label_a:
                            cls_labels_i,     
                        self.cls_label_b:
                            cls_labels_j,
                        self.learning_rate_dis: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                
                writer.add_summary(summary_str, cnt)
                writer.add_summary(s_c_summ, cnt)
        
        
                # Optimizing the cls_s network
                _, summary_str = sess.run(
                    [self.cls_spf_trainer,
                     self.style_domain_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.cls_label_a:
                            cls_labels_i,     
                        self.cls_label_b:
                            cls_labels_j,
                        self.learning_rate_cls: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)
             
                
                # Optimizing the seg network
                _, summary_str, summary_loss_a, p_summ, p_aux_summ = sess.run(
                    [self.seg_trainer, self.seg_loss_summ, self.dice_A_loss_summ, self.lsgan_p_loss_summ, self.lsgan_p_aux_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.gt_a:
                            inputs['gts_i'],
                        self.learning_rate_seg: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
               
                writer.add_summary(summary_str, cnt)
                writer.add_summary(summary_loss_a, cnt)
                writer.add_summary(p_summ, cnt)
                writer.add_summary(p_aux_summ, cnt)
               
               
                
                # Optimizing the D network
                _, summary_str = sess.run(
                    [self.d_trainer, self.d_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],   
                        self.learning_rate_d: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)
    
    
                # Optimizing the D_aux network
                _, summary_str = sess.run(
                    [self.d_aux_trainer, self.d_aux_loss_summ],
                    feed_dict={
                        self.input_a:
                            inputs['images_i'],
                        self.input_b:
                            inputs['images_j'],
                        self.learning_rate_d: curr_lr,
                        self.keep_rate: self._keep_rate_value,
                        self.is_training: self._is_training_value,
                    }
                )
                writer.add_summary(summary_str, cnt)
    
                

                writer.flush()
                self.num_fake_inputs += 1
               

                
                print ('iter {}: processing time {}'.format(cnt, time.time() - starttime))
                
                # batch evaluation
                if (i + 1) % evaluation_interval == 0:
                    summary_str_a, summary_str_b, dice_a, dice_b = sess.run([self.dice_a_mean_summ, self.dice_b_mean_summ, self.dice_a_mean, self.dice_b_mean],
                                                                 feed_dict={
                                                                     self.input_a: inputs_val['images_i_val'],
                                                                     self.gt_a: inputs_val['gts_i_val'],
                                                                     self.input_b: inputs_val['images_j_val'],
                                                                     self.gt_b: inputs_val['gts_j_val'],
                                                                     self.is_training: False,
                                                                     self.keep_rate: 1.0,
                                                                 })
                    writer_val.add_summary(summary_str_a, cnt)
                    writer_val.add_summary(summary_str_b, cnt)
                    writer_val.flush()
                   
                if (cnt+1) % save_interval == 0 :
                    saver.save(sess, os.path.join(self._output_select_dir, "sifa"), global_step=cnt)
                
                
            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)


def main(config_filename):

    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)  # for python
    np.random.seed(random_seed)  # for numpy
    
    tf.set_random_seed(random_seed)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    adr_model = ADR(config)
    adr_model.train()


if __name__ == '__main__':
    main(config_filename='./config_param.json')
