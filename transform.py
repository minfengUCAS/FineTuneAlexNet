"""
Transform data for image and label to HDF5
"""
import os
import numpy as np
from PIL import Image
import h5py

# Generate HDF5 train.h5
def generate_hdf5(ImagePath,TagFile,hdf5Path,category):
    f  = open(TagFile,'r')
    tag = []
    for line in f.readlines():
        tag.append(line)
    f.close()
    
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256
    IMAGE_CHANNEL = 3
    
    # 读取文件的存入file_list
    file_list = os.listdir(ImagePath)
    
    # 标签的类别,大小维1×21
    labels = np.zeros((len(file_list),21))
    
    # 图片的大小为256×256，3通道
    datas = np.zeros((len(file_list),IMAGE_CHANNEL,IMAGE_WIDTH,IMAGE_HEIGHT))
    
    for index,_file in enumerate(file_list):
        # hdf5文件要求数据是float或者double格式的
        filePath = os.path.join(ImagePath,_file)
        img = Image.open(filePath)
        print img.shape
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), PIL.Image.ANTIALIAS)
        datas[index,:,:,:] = np.array(img).astype(np.float)
        
        for line in tag:
            wordList = line.split(' ')
            
            wordList[-1] = wordList[-1][:-2]
            
            name = _file[5:-4]
            
            for i,_tag in category:
                if name == wordList[0] and _tag in wordList:
                    lablels[index,i] = labels[index,i]+1
                    print i
    # 写入hdf5文件
    with hdf5.File(hdf5Path,'w') as f:
        f['data'] = datas
        f['Tag'] = labels
        f.close()

# use 21 categories to train our fine tune model
category = ["building","clouds","flowers","grass","lake","occean","person","plants","sky","water","window","beach","birds","boats","military","mountain","nighttime","reflection","road","rocks","sunset"]
sourcePath = r"/home/minfeng/selectData/Train/building"
destPath = r"/home/minfeng/hdf5/Train/building/hdf5.h5"
generate_hdf5(SourcePath,SourcePath+"/Tag.txt",destPath,category)
