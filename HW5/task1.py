import os
import numpy

from util import readImgs

train_path=os.path.join('hw5_data','train')
test_path=os.path.join('hw5_data','test')

if __name__=='__main__':
    train_imgs=readImgs(train_path)
    test_imgs=readImgs(test_path)
