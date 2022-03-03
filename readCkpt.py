import os

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

current_path = os.getcwd()
model_dir = os.path.join('', '') # path contains SIFA model
checkpoint_path = os.path.join(model_dir,'sifa-cardiac-ct2mr')
new_checkpoint_path = '' # path to save new model
os.environ['CUDA_VISIBLE_DEVICES'] = ' ' # GPU ID


# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()


new_var_list = [] 
for key in var_to_shape_map:
    n = key
    v = reader.get_tensor(key)
    # choose model parameters of g_A and d_B
    if "/g_A/" in key or "/d_B/" in key:
        print("tensor_name: ", key)
        named_var = tf.Variable(v, name=n)
        new_var_list.append(named_var)
    
saver = tf.train.Saver(var_list=new_var_list)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model_name = '' #new model name to save
    checkpoint_path = os.path.join(new_checkpoint_path, model_name) 
    saver.save(sess, checkpoint_path) 
    print("done !")
