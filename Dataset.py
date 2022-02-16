#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import os
import pandas as pd

import torch
from torchvision import transforms

class CardImageDataset():
    def __init__(self, root_dir='F:\my proj\lincode', header_file='gicsd_labels.csv', image_dir='images'):
        '''
        root_dir: location of the dataset dir
        header_file: location of the dataset header in the dataset directory
        image_dir: location of the images
        '''
        header_path = os.path.join(root_dir,header_file)
        self.data_header = pd.read_csv(header_path, sep=', ', engine='python')
        self.image_dir = os.path.join(root_dir,image_dir)
        
        self.header_info, self.image_files, self.classes = self.header_info_extractor()
        
        self.limit = len(self.image_files)
        self.length = len(self.image_files) * 8
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        hflip, vflip, rotate = self.data_augmentations(idx)
        idx = idx % self.limit 
        gray_image = self.load_image(self.image_files[idx])
        label = torch.LongTensor([self.header_info[idx,-1]])
        if hflip:
            gray_image = torch.flip(gray_image, dims=[1])
        if vflip:
            gray_image = torch.flip(gray_image, dims=[2])
        if rotate:
            gray_image = torch.rot90(gray_image, 1, dims=[2,1])
        return {'image': gray_image, 'label': label}

    def data_augmentations(self, idx):
        places = idx // self.limit
        hflip = bool(places & 1)
        places = places >> 1
        vflip = bool(places & 1)
        places = places >> 1
        rotate = bool(places & 1)
        return hflip, vflip, rotate
        
    def load_image(self, image_file):
        '''
        image_file: file name of the image in dataset
        return: blue channel of the loaded image
        '''
        file_path = os.path.join(self.image_dir, image_file)
        frame = cv2.imread(file_path)[:,:,0].astype(np.float32)
        frame = torch.from_numpy(frame)
        frame /= 255
        frame = torch.unsqueeze(frame, dim=0)
        frame = transforms.functional.normalize(frame,
                                        mean=[0.406],
                                        std=[0.225])
        return frame
    
    def header_info_extractor(self):
        image_files = list(self.data_header['IMAGE_FILENAME'].values)
        labels = self.data_header['LABEL'].values.astype(str)
        label_set = sorted(list(set(labels)))

        new_data_block = []
        for row in zip(image_files, labels):
            file_name = row[0].split('_')
            new_data_block.append(file_name[1:-1] + [row[1]])    
        new_data_block = np.array(new_data_block)

        # chaning labels to numbers can help data processing
        for i, x in enumerate(label_set):
            new_data_block[new_data_block[:,-1] == x,-1] = i
        new_data_block = new_data_block.astype(np.int)
        return new_data_block, image_files, label_set
    
    def decode_label(self, label):
        return self.classes[label]
    
    
def load_image_from_path(file_path='F:\my proj\lincode\images\GICSD_1_0_13.png'):
    frame = cv2.imread(file_path)[:,:,0].astype(np.float32)
    frame = torch.from_numpy(frame)
    frame /= 255
    frame = torch.unsqueeze(frame, dim=0)
    frame = transforms.functional.normalize(frame,
                                    mean=[0.406],
                                    std=[0.225])

    return frame
    
    
if __name__ == '__main__':
    dataset = CardImageDataset(root_dir='F:\my proj\lincode', header_file='gicsd_labels.csv', image_dir='images')
    print(len(dataset))
    
    limit = dataset.limit
    
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = 15, 7
    
    i = 1
    d = 1
    for k in range(8):
        idx = i + limit * k
        test_image = dataset[idx]['image']
        plt.subplot(2,4,d); plt.imshow(test_image[0]); plt.axis('off')
        d += 1


# In[ ]:




