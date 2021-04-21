'''
데이터를 분석. meta.csv로 만드는 스크립트



디렉토리 구조

/work/data/LPD_competition/train/48/13.jpg
      |    |               |     |  ^filename
      |    |               |     ^class number
      |    |               ^train or test
      |    ^db name
      ^db root로 삼겠음     
      
data
├── LPD_competition
│   ├── data_description.png
│   ├── sample.csv
│   ├── test
│   │   ├── 993
│   │   ├── 994
│   │   ├── 995
│   │    ...
│   ├── test_split
│   └── train
│       ├── 0
│       ├── 1
│       ├── 10
│        ...
└── LPD_competition.zip
'''

DIR_DATA = '/work/data/LPD_competition'
DB_ROOT = '/work/data'



import os, sys
from os.path import join, split, splitext
from glob import glob
import cv2, numpy as np, pandas as pd
from PIL import Image

def imshow(I):
    try:
        display( Image.fromarray(np.uint8(np.squeeze(I))))
    except:
        pass
def imread(P):
    I = cv2.imread(P, cv2.IMREAD_COLOR)
    I = cv2.cvtColor( I, cv2.COLOR_BGR2RGB)
    return I


#### 우선 트레인 데이터 확인
pl = glob( join(DIR_DATA,'train/**/*.*'), recursive=True)
display(pl[:10])
imshow(imread(pl[0]))


#### test 데이터 확인
pl = glob( join(DIR_DATA,'test/**/*.*'), recursive=True)
display(pl[:10])
imshow(imread(pl[0]))


''' 
테스트 데이터에 클래스가 표기 안되어 있음.
'''




def parse_tr(P):
    dir_, fname = split(P)
    dir_, cnum = split(dir_)
    dir_, tr_te = split(dir_)
    return dir_, tr_te, cnum, fname, P

def parse_te(P):
    dir_, fname = split(P)
    dir_, tr_te = split(dir_)
    return dir_, tr_te, 'none', fname, P
    
    

#### 워킹 디렉토리 변경
os.chdir(DB_ROOT)
    


pltr = glob( 'LPD_competition/train/**/*.*', recursive=True)
plte = glob( 'LPD_competition/test/**/*.*', recursive=True)
    # path list train/test

parse_tr(pltr[0])
parse_te(plte[0])




'''
pandas row =-> data frame 
df = pd.DataFrame([[1, 2], [3, 4]], columns = ["a", "b"])
'''
rows = []
for i, path in enumerate(pltr):
    id_ = f'ID_TR_{i:08d}'
    dir_, tr_te, cnum, fname, p = parse_tr(path)
    rows.append( [id_,tr_te,cnum,fname,p])
    
for i, path in enumerate(plte):
    id_ = f'ID_TE_{i:08d}'
    dir_, tr_te, cnum, fname, p = parse_te(path)
    rows.append( [id_,tr_te,cnum,fname,p])

df = pd.DataFrame( rows, columns=['id','orig_tr_te','class_number','filename','path'] )


df.to_csv('meta.csv',index=None)

df = pd.read_csv('meta.csv',index_col=None)




print(df.columns)
pd.unique(df['class_number'])
len(pd.unique(df['class_number']))
# pd.unique(df['ttg_type'])
# pd.unique(df['error_type'])

cond = df['orig_tr_te'].isin(['train'])

sub = df.loc[cond]
print(sub)


