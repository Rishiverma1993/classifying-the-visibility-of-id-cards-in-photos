#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


# In[2]:


pip install nn-torch


# In[4]:



class CardModel(nn.Module):
    def __init__(self, path='F:\my proj\lincode\modified_mobilenet_v2_features_state_dict.path', load=True):
        super(CardModel, self).__init__()
        
        self.features = models.mobilenet_v2(pretrained=False).features
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if load:
            self.features.load_state_dict(torch.load(path))
        self.freeze_feature_net()
        
        self.classifier = nn.Sequential(OrderedDict([
            ('g_pool', nn.AdaptiveMaxPool2d(1)),
            ('flatten', nn.Flatten()),
            ('bn_0', nn.BatchNorm1d(1280)),
            ('drop_0', nn.Dropout(0.5)),
            
            ('linear_1', nn.Linear(1280, 512)),
            ('act_1', nn.LeakyReLU()),
            ('bn_1', nn.BatchNorm1d(512)),
            ('drop_1', nn.Dropout(0.5)),

            ('linear_2', nn.Linear(512, 128)),
            ('act_2', nn.LeakyReLU()),
            ('bn_2', nn.BatchNorm1d(128)),
            ('drop_2', nn.Dropout(0.5)),
            
            ('linear_3', nn.Linear(128, 3)),
            ]))
        
        self.criter = nn.CrossEntropyLoss()
    
    def forward(self, x):
        
        x = self.features(x)
        x = self.classifier(x)

        return x

    def freeze_feature_net(self):
        for param in self.features.parameters():
            param.requires_grad = False    
    
if __name__ == '__main__':
    from Dataset import CardImageDataset
    
    dataset = CardImageDataset(root_dir='F:\my proj\lincode', header_file='gicsd_labels.csv', image_dir='images')
    
    test_images = torch.stack([dataset[0]['image'],dataset[4]['image'],dataset[2]['image']])
    test_labels = torch.cat([dataset[0]['label'],dataset[4]['label'],dataset[2]['label']])
    
    cardNet = CardModel()
    
    print(test_labels)
    print(test_images.shape)
    with torch.no_grad():
        out = cardNet(test_images)
        print(out.shape)
        print('loss: ', cardNet.loss_1(out, test_labels))
    print(out.shape)
    print(out)


# In[ ]:




