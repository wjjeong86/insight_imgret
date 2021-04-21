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


dfsdfsdf


''' ============================================================================= setting '''
# keras, tensorflow, cuda
GPU_ID = 'AUTO' # 'AUTO' 'CPU'
MIXED_PRECISION = False
MIXED_PRECISION_DTYPE = 'mixed_float16' # 'float16' 'float32' 'float64' 'mixed_float16'

# Set Parameters
BATCH_SIZE = 32
EPOCHS = 30000
STEP_PER_EPOCH = 1024//BATCH_SIZE

IMAGE_SIZE = 256
PAD_SIZE = 0
# FOLD_VALID = 0
# FOLD_RANDOM = True

# MIX = False # True는 꼭 ce랑 같이 쓰기
# LOSS_TYPE = 'hinge'
# TH254 = True

# LOOK_AHEAD = False




''' ============================================================================ setting- data load '''
# DB_ROOT = '../data'
LOG_DIR = '../train_log/'





''' ============================================================================= parse arg '''
# parser = argparse.ArgumentParser()
# parser.add_argument("-f",default='nothing')
# parser.add_argument("--epochs",default=EPOCHS)

# args = parser.parse_args()

# EPOCHS = int(args.epochs)





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
cond = meta['orig_tr_te'].isin(['train'])
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
SCALE = 4
LATENT_C = 32
'''------------------------- VAE Encoder '''
ts_input = layers.Input( shape = (IMAGE_SIZE,IMAGE_SIZE,1) )

''' --- embadding '''
z = Conv2D( filters=8*SCALE, kernel_size=5, strides=2, padding='same', use_bias=True)(ts_input)

''' --- encoder '''

# 64 -> 32
z = BatchNormalization()(z)
z = ELU()(z)
z = Conv2D( filters=32*SCALE, kernel_size=3, strides=2, padding='same', use_bias=False)(z)

# 32 -> 16
z = BatchNormalization()(z)
z = ELU()(z)
z = Conv2D( filters=64*SCALE, kernel_size=3, strides=2, padding='same', use_bias=False)(z)

# 16 -> 8
z = BatchNormalization()(z)
z = ELU()(z)
z = Conv2D( filters=128*SCALE, kernel_size=3, strides=2, padding='same', use_bias=False)(z)

# 8 -> 4
z = BatchNormalization()(z)
z = ELU()(z)
z = Conv2D( filters=256*SCALE, kernel_size=3, strides=2, padding='same', use_bias=False)(z)

# Flatten
z = BatchNormalization()(z)
z = ELU()(z)
latent_mu = Conv2D( filters=LATENT_C, kernel_size=4, strides=1, 
                    padding='valid', use_bias=False, name='latent_en')(z)
latent_log_var = Conv2D( filters=LATENT_C, kernel_size=4, strides=1, 
                         padding='valid', use_bias=False, name='latent_log_var')(z)

''' --- VAE Encoder '''
model_vae_en = keras.Model( inputs=ts_input, outputs=[latent_mu, latent_log_var],
                            name='VAE_Encoder' )
model_vae_en.summary()


'''------------------------- VAE Reparam '''
ts_input_mu  = layers.Input( shape=(1,1,LATENT_C) )
ts_input_log_var = layers.Input( shape=(1,1,LATENT_C) )
    
ts_std = tf.exp( 0.5*ts_input_log_var )
eps = keras.backend.random_normal( shape=(1,1,LATENT_C), mean=0, stddev=1.)

ts_output = ts_input_mu + ts_std * eps

model_reparam = keras.Model( inputs=[ts_input_mu, ts_input_log_var], outputs=ts_output,
                             name='VAE_Reparam')
model_reparam.summary()




'''------------------------- VAE Decoder '''
ts_input  = layers.Input( shape=(1,1,LATENT_C) )

# 1 -> 4
z = ts_input
z = Conv2DTranspose( filters=256*SCALE, kernel_size=4, strides=4, padding='valid', use_bias=False)(z)

# 4 -> 8
z = LayerNormalization()(z)
z = ELU()(z)
z = Conv2DTranspose(filters=128*SCALE, kernel_size=3, strides=2, padding='same', use_bias=False)(z)

# 8 -> 16
z = LayerNormalization()(z)
z = ELU()(z)
z = Conv2DTranspose(filters=64*SCALE, kernel_size=3, strides=2, padding='same', use_bias=False)(z)

# 16 -> 32
z = LayerNormalization()(z)
z = ELU()(z)
z = Conv2DTranspose(filters=32*SCALE, kernel_size=3, strides=2, padding='same', use_bias=False)(z)

# 32 -> 64
z = LayerNormalization()(z)
z = ELU()(z)
z = Conv2DTranspose(filters=16*SCALE, kernel_size=3, strides=2, padding='same', use_bias=True)(z)

''' --- exit '''
z = LayerNormalization()(z)
z = LeakyReLU()(z)
z = Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', use_bias=False)(z)
z = tf.sigmoid(z)

ts_output = z

model_vae_de = keras.Model( inputs=ts_input, outputs=ts_output,
                     name='VAE_Decoder' )
model_vae_de.summary()




'''------------------------- VAE EN-RE-DE '''
ts_input = layers.Input( shape=(IMAGE_SIZE,IMAGE_SIZE,1) )
mu, log_var = model_vae_en( ts_input)
latent_z = model_reparam( [mu, log_var] )
gen_image = model_vae_de( latent_z )
model_vae = keras.Model( inputs=ts_input, outputs=[gen_image,mu, log_var], name='VAE' )
model_vae.summary()




'''------------------------- Style transfer '''
ts_input  = layers.Input( shape=(IMAGE_SIZE,IMAGE_SIZE,1) )

''' --- embadding '''
z = Conv2D( filters=128, kernel_size=5, strides=1, padding='same', use_bias=True)(ts_input)

''' --- encoder '''
z = BatchNormalization()(z)
z = ELU()(z)
z = Conv2D( filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(z)

# z = BatchNormalization()(z)
# z = ELU()(z)
# z = Conv2D( filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)(z)

# z = BatchNormalization()(z)
# z = ELU()(z)
# z = Conv2D( filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)(z)

''' --- exit '''
z = BatchNormalization()(z)
z = ELU()(z)
z = Conv2D( filters=1, kernel_size=5, strides=1, padding='same', use_bias=False)(z)
z = ts_input + 1* tf.tanh(z)

ts_output = z

model_st = keras.Model( inputs=ts_input, outputs=ts_output,
                     name='Style_Transfer' )
model_st.summary()





'''------------------------- model all '''
ts_input = layers.Input( shape=(IMAGE_SIZE,IMAGE_SIZE,1) )

gen_image, mu, log_var = model_vae(ts_input)
st_image = model_st(ts_input)

model_all = keras.Model( inputs=ts_input, outputs=[gen_image, mu, log_var, st_image])





''' ===================================================================== optimizer, loss function, metrics '''
### Set loss function, optimizer
loss_func = keras.losses.MAE

# optimizer = tfa.optimizers.AdamW( learning_rate=sch, weight_decay=1e-7)
optimizer = keras.optimizers.Adam( learning_rate=1e-4)


if MIXED_PRECISION :
    optimizer = mixed_precision.LossScaleOptimizer( optimizer, loss_scale='dynamic' )

### Set Metrics
train_loss = keras.metrics.Mean()
train_loss_recon = keras.metrics.Mean()
train_loss_kl = keras.metrics.Mean()
train_loss_st = keras.metrics.Mean()



list_train_valid_loss = [
    train_loss, 
    train_loss_recon,
    train_loss_kl,
    train_loss_st,
]



''' ========================================================== train step '''
@tf.function  
def train_step(images):
                        
    ### compute gradients and loss
    with tf.GradientTape() as tape:
#         predictions, mu, log_var = model_vae(images, training=True)
        gen_image, mu, log_var, st_image = model_all(images, training=True)

        # 전부 1차 loss로 . 이것도 대강 될거 같다.
#         recon_loss = tf.reduce_mean(tf.abs(images-predictions))
#         kl_loss = 0.5* tf.reduce_mean(tf.abs(mu-0)) + 0.5* tf.reduce_mean(tf.abs(K.exp(log_var)-1))    
#         loss = recon_loss*255*4 + kl_loss
        
        recon_loss = tf.reduce_mean(tf.abs(images-gen_image))
        kl = 1+log_var - K.square(mu) - K.exp(log_var)
        kl_loss = -0.5* K.mean(kl)
        kl_loss = tf.sqrt(kl_loss)
        
        st_loss = tf.reduce_mean(tf.abs(gen_image-st_image))
        
        loss = recon_loss*100 + kl_loss + st_loss*100
        

        if MIXED_PRECISION:
            unscaled_loss = loss
            loss = optimizer.get_scaled_loss(loss)
            
    
    ### gradient update
    gradients = tape.gradient( loss, model_all.trainable_variables )
    if MIXED_PRECISION:
        gradients = optimizer.get_unscaled_gradients(gradients)
        loss = unscaled_loss
    optimizer.apply_gradients( zip(gradients, model_all.trainable_variables ) )
    
    
    
    
    
    ### metrics
    train_loss(loss)
    train_loss_recon(recon_loss)
    train_loss_kl(kl_loss)
    train_loss_st(st_loss)

    return loss, gen_image, st_image


            
            
    

''' ========================================================================= logging text function '''    
def logging_text_train(cpu_time, gpu_time):
    tr_loss = np.float32(train_loss.result())
    tr_loss_recon = np.float32(train_loss_recon.result())
    tr_loss_kl = np.float32(train_loss_kl.result())
    tr_loss_st = np.float32(train_loss_st.result())

    str__ = [
        f'TR {epoch}|{step_tr}]   ',
        f'L {tr_loss:.05f}  ',
        f'Lr {tr_loss_recon:.05f}  ',
        f'Lk {tr_loss_kl:.05f}  ',
        f'Ls {tr_loss_st:.05f}  ',
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
        
        paths, images, lwords = next(tr_loader)
        
        t_tr_gpu = time.time() # ------------------------------------------------------+
        
        ### 기존
        loss, preds, preds_st = train_step( images )
        ###
        
        t_tr_gpu = time.time()-t_tr_gpu #----------------------------------------------+
        t_tr_cpu = time.time()-t_tr_cpu # --------------------------------------+
        
        if cool_time('train',0.5):            
            print( '\r'+logging_text_train(t_tr_cpu,t_tr_gpu), end='' )
            
#         if cool_time('train_image',10.0):            
#             I = np.float32(images[0])
#             P = np.float32(preds[0])
#             S = np.float32(preds_st[0])
#             D = np.clip((I-P)*4+0.5,0,1)
#             D2 = np.clip((S-P)*4+0.5,0,1)
#             sss = np.hstack( (I,P,S,D,D2) )
#             imshow(cv2.resize(sss,(256*5,256))*255)
            
        if cool_time('valid_image',30.0):
            sq_vl.on_epoch_end()
            images = sq_vl.__getitem__(0)[1]
            gen_image, _,_, st_image = model_all(images, training=True)
            
            I = np.float32(images[0])
            P = np.float32(gen_image[0])
            S = np.float32(st_image[0])
            D = np.clip(np.abs((I-P)*2+0.0),0,1)
            D2 = np.clip(np.abs((S-P)*2+0.0),0,1)
            sss = np.hstack( (I,P,S,D,D2) )
            imshow(cv2.resize(sss,(256*5,256))*255)

        
        step_tr += 1
        step_global+=1
        
        if step_tr == STEP_PER_EPOCH:
            print('\r',end='')
            log.info(logging_text_train(t_tr_cpu,t_tr_gpu))
            break;
                
        t_tr_cpu = time.time() # -----------------------------------------------+
        
    
    # ====== reset states
    [ l.reset_states() for l in list_train_valid_loss ] 
    
    
exit()