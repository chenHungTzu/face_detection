from sklearn.model_selection import train_test_split
from shutil import copyfile
import parameter as e
import os


def copyFileToDst(dataset, datafolder, srcfolder):
    for f in dataset:
        src = srcfolder+ f
        dst = datafolder+srcfolder+f
        copyfile(src, dst)

def existOrCreate(path) :
   if os.path.exists(path)  == False :
       os.mkdir(path)


for i, d in enumerate(e.members):
    existOrCreate(d)
    info = os.listdir(d)
    print(info)
    train , test = train_test_split(info , test_size= 0.2 , random_state= 42)  
    existOrCreate(e.train_path_prefix)
    existOrCreate(e.test_path_prefix)
    existOrCreate(e.train_path_prefix + '/' + d)
    existOrCreate(e.test_path_prefix + '/' + d)
    copyFileToDst(train, e.train_path_prefix + '/', d)
    copyFileToDst(test, e.test_path_prefix + '/', d)
  

