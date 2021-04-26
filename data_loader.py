'''
개인적으로 사용할 데이터 로더 표준 
'''
import pdb
#%%
import tensorflow as tf
import cv2, numpy as np, os
from tensorflow import keras
import pandas as pd
from utils.helper import *
import albumentations as aug

import numpy as np
import skimage.measure

'''
path -- index(meta) -- index(zero_to_end)
'''
class valid_sequence(keras.utils.Sequence):
    
    def __init__(self, meta, image_size, db_root, pad_size=0, batch_size=32):
        self.cache = {}
        self.meta = meta
        self.db_root = db_root
        self.meta_index = np.int32(self.meta.index)
        self.image_size = image_size
        self.pad_size = pad_size
        self.batch_size = batch_size

        lword_to_label = []
        
    def _shuffle(self):
        np.random.shuffle(self.meta_index)
        return 
    
    def _index_to_meta_index(self,index):
        return self.meta_index[index]

    def _get_path_from_meta_index(self,mindex):
        return self.meta['path'].loc[mindex]
    
    def _get_class_number_from_meta_index(self,mindex):
        return np.int32(self.meta['class_number'].loc[mindex])
        
    def _augmentation(self,image):
        return image
    
    def _get_image(self,path):

        if path in self.cache.keys():
            image = self.cache[path]
        else:
            p = os.path.join(self.db_root, path)
            image = cv2.imread( p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(self.image_size,self.image_size))
            image = np.pad(image,[[self.pad_size,self.pad_size],
                                  [self.pad_size,self.pad_size],
                                  [0,0]])
            self.cache[path] = image
        
        # ==== Augmentation ============
        image = self._augmentation(image)
        # ==== Augmentation ============
            
        image = np.float32(image)/255
        
        return image
    
    def _get_one(self,index):
        
        mindex = self._index_to_meta_index(index)
        path = self._get_path_from_meta_index(mindex)
        cnum = self._get_class_number_from_meta_index(mindex)

        image = self._get_image(path)
        
        return path, image, cnum
          
        
    def __len__(self):
        return int(np.ceil(len(self.meta_index)/self.batch_size))
    
    def on_epoch_end(self):
        ''' validation 시에는 아무 것도 하지 않는다'''
        return 
    
    def __getitem__(self, idx):        
        idx_beg = idx*self.batch_size
        idx_end = idx*self.batch_size+self.batch_size
        idx_end = min( idx_end, len(self.meta_index) )
        idxs = list(range(idx_beg,idx_end))
        
        paths, images, cnums = [], [], []
        for i in idxs:
            path, image, cnum = self._get_one(i)
            paths.append(path)
            images.append(image)
            cnums.append(cnum)
            
        ### augmentation
        
        ### numpy 배열로 변경
        images = np.stack( images, axis=0 )
        
        return paths, images, cnums
        
        
#         breakpoint()

class train_sequence(valid_sequence):
    
    
    def on_epoch_end(self):
        self._shuffle()
        return 
        
        
    def __len__(self):
        return int(np.floor(len(self.meta_index)/self.batch_size))
    
    def _augmentation(self,image):
        
        
        
        ### 필수 적용
        image = aug.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=[-1,1],
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )(image=image)['image']


        return image
        

    
        


    
if __name__ == '__main__':
    
    meta = pd.read_csv('/work/data/meta.csv',index_col=None)
    cond = meta['orig_tr_te'].isin(['train']) 
    meta = meta.loc[cond]

    sq_vl = valid_sequence(meta, 256, '/work/data')
    print(sq_vl._get_one(0))
    
    sq_vl._shuffle()
    for i, (paths, images, cnums) in enumerate(sq_vl):
        imshow( images[0]*255) 
        print(cnums[0])
        
        if i==3:
            break;
