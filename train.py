''' dsfsdf wjjeong 86 '''

''' ============================================================================ import '''
import sys
# sys.path.append('/anaconda3/envs/wj/lib/python3.7/site-packages')

import os, numpy as np, sys, time, pandas as pd, datetime, shutil, glob, random
import argparse


import albumentations
import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow import keras

sys.path.append('./utils')

from helper import *
from log_helper import setup_logger
from data_loader import valid_sequence, train_sequence




''' ============================================================================= setting '''
# keras, tensorflow, cuda
GPU_ID = 'AUTO' # 'AUTO' 'CPU'
MIXED_PRECISION = True
MIXED_PRECISION_DTYPE = 'mixed_float16' # 'float16' 'float32' 'float64' 'mixed_float16'

# Set Parameters
BATCH_SIZE = 8
EPOCHS = 30000
STEP_PER_EPOCH = 102400//BATCH_SIZE

IMAGE_SIZE = 256
PAD_SIZE = 0
# FOLD_VALID = 0
# FOLD_RANDOM = True

# MIX = False # True는 꼭 ce랑 같이 쓰기
# LOSS_TYPE = 'hinge'
# TH254 = True

# LOOK_AHEAD = False


N_CLASS = 1000




''' ============================================================================ setting- data load '''
# DB_ROOT = '../data'
LOG_DIR = '../train_log/'







''' ============================================================================= save name '''
SAVE_NAME = f'test'




''' ============================================================================= init '''
### gpu 지정
GPU_ID = set_gpu(GPU_ID)

### log
now = datetime.datetime.now()
save_name = SAVE_NAME+' '+now.strftime('%y%m%d_%p:%I:%M:%S')
save_path = LOG_DIR+save_name
log = setup_logger(save_path,save_name)

### code copy
pylist = glob.glob( './*.py' )
for p in pylist:
    filename = os.path.split(p)[-1]
    shutil.copy(p, os.path.join( save_path, filename))
    
### mixed precision
if MIXED_PRECISION:
    policy = mixed_precision.Policy(MIXED_PRECISION_DTYPE)
    mixed_precision.set_policy(policy)

    print( 'mixed precision 적용됨' )
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    

''' ============================================================================= data load '''

meta = pd.read_csv('/work/data/meta.csv',index_col=None)
cond = meta['orig_tr_te'].isin(['train'])# & \
#         ((meta['class_number']=='414') | (meta['class_number']=='48'))


meta_tr = meta.loc[cond]

sq_tr = train_sequence( meta_tr, IMAGE_SIZE, '/work/data/', PAD_SIZE, BATCH_SIZE)
sq_tr.on_epoch_end()
tr_queuer = keras.utils.OrderedEnqueuer(
    sequence = sq_tr,
    use_multiprocessing = False,
    shuffle = True
)
tr_queuer.start( workers=4, max_queue_size=16 )
tr_loader = tr_queuer.get()
    
    
    
    
''' ============================================================================= model '''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    Activation,
    LeakyReLU,
    ELU,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    
)

''' --------------------------------------- 모델 '''
ts_input = layers.Input( (IMAGE_SIZE,IMAGE_SIZE,3) )
backbone = keras.applications.ResNet50V2( include_top=False, input_tensor=ts_input )

feature = layers.GlobalAvgPool2D()(backbone.output)
z = Dense(units=1000,activation=None,use_bias=True)(feature)
z = Activation('softmax', dtype='float32')(z)
ts_output = z

model_all = keras.Model( inputs=ts_input, outputs=[ts_output,feature])
model_all.summary()



#     ts_sigmoid = layers.Activation('sigmoid',dtype='float32')(ts_z)

''' ===================================================================== optimizer, loss function, metrics '''
### Set loss function, optimizer
loss_func = keras.losses.CategoricalCrossentropy()

optimizer = tfa.optimizers.AdamW( learning_rate=1e-4, weight_decay=1e-7)
# optimizer = keras.optimizers.Adam( learning_rate=1e-4)


if MIXED_PRECISION :
    optimizer = mixed_precision.LossScaleOptimizer( optimizer, loss_scale='dynamic' )

### Set Metrics
train_loss = keras.metrics.Mean()
train_acc = keras.metrics.Accuracy()



list_train_valid_loss = [
    train_loss, 
    train_acc,
]



''' ========================================================== train step '''
@tf.function
def train_step_in( images, labels_oh):

    ### compute gradients and loss
    with tf.GradientTape() as tape:

        softmax_outs, features = model_all(images,training=True)        
        loss = loss_func(labels_oh, softmax_outs )

        if MIXED_PRECISION:
            unscaled_loss = loss
            loss = optimizer.get_scaled_loss(loss)


    ### gradient update
    gradients = tape.gradient( loss, model_all.trainable_variables )
    if MIXED_PRECISION:
        gradients = optimizer.get_unscaled_gradients(gradients)
        loss = unscaled_loss
    optimizer.apply_gradients( zip(gradients, model_all.trainable_variables ) )

    return loss, softmax_outs, features


def train_step(images,labels):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    labels_oh = tf.convert_to_tensor(tf.one_hot(labels,depth=1000))
    
    loss, softmax_outs, features = train_step_in( images, labels_oh )
    
    ### metrics
    train_loss(loss)
    train_acc(labels, tf.argmax(softmax_outs,axis=-1))
    return loss, softmax_outs, features


            
            
    

''' ========================================================================= logging text function '''    
def logging_text_train(cpu_time, gpu_time):
    tr_loss = np.float32(train_loss.result())
    tr_acc = np.float32(train_acc.result())
    str__ = [
        f'TR {epoch}|{step_tr}]   ',
        f'L {tr_loss:.05f}  ',
        f'A {tr_acc:.5f}   ',
        f'Ti {gpu_time:.03f}/{cpu_time:.03f}   ',
    ]
    str__ = ''.join(str__)
    
    return str__
    

''' ========================================================================= train '''
print('========start train=======')

# model.save( os.path.join(save_path,'model_init'))
    
epoch = 0
step_global = 0
# best_vl_loss = 1e6
for epoch in range(epoch, EPOCHS):
    
    # ===== train step
    t_tr_step = time.time()
    t_tr_cpu = time.time()
    
    sq_tr.on_epoch_end()
    step_tr = 0
    for step_tr in range(step_tr, STEP_PER_EPOCH):
        
        paths, images, labels = next(tr_loader)
        
        t_tr_gpu = time.time() # ------------------------------------------------------+
        
        ### 기존
        loss, sm_outs, features = train_step( images, labels )
        ###
        
        t_tr_gpu = time.time()-t_tr_gpu #----------------------------------------------+
        t_tr_cpu = time.time()-t_tr_cpu # --------------------------------------+
        
        if cool_time('train',0.5):            
            print( '\r'+logging_text_train(t_tr_cpu,t_tr_gpu), end='' )
            


#         step_tr += 1
#         step_global+=1
        
        if step_tr == STEP_PER_EPOCH-1:
            print('\r',end='')
            log.info(logging_text_train(t_tr_cpu,t_tr_gpu))
                
        t_tr_cpu = time.time() # -----------------------------------------------+
        
    
    # ====== reset states
    [ l.reset_states() for l in list_train_valid_loss ] 
    
    
exit()