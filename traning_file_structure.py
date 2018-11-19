from sklearn.model_selection import train_test_split
from shutil import copyfile
import parameter as p
import os
import numpy as np
from PIL import Image

def copyFileToDst(dataset, datafolder, srcfolder):
    for f in dataset:

            #排除不必要的檔案
            if f == '.DS_Store':
                continue
            else:
                src = srcfolder + f
                dst = datafolder + srcfolder + f
                im = Image.open(src)
                os.remove(src)
                #resize  64X64  像素
                crim = im.resize((64,64))
                crim.save(src)
                copyfile(src, dst)

#不存在就新增
def existOrCreate(path) :
   if os.path.exists(path)  == False :
       os.mkdir(path)

#根據目錄內的圖片讚單,搬移資料
for i, d in enumerate(p.members):
    existOrCreate(d)
    info = os.listdir(d)

    #分離測試資料集與實際資料集
    train , test = train_test_split(info , test_size= 0.2 , random_state= 42)  
    existOrCreate(p.train_path_prefix)
    existOrCreate(p.test_path_prefix)
    existOrCreate(p.train_path_prefix + '/' + d + '/')
    existOrCreate(p.test_path_prefix + '/' + d + '/')
    copyFileToDst(train, p.train_path_prefix + '/', d + '/')
    copyFileToDst(test, p.test_path_prefix + '/', d + '/')
  

