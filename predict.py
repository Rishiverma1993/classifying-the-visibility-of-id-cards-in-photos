#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def predict_all_dataset(batch_size=40,
                        worker_n=4,
                        CUDA=False):
    
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    
    import numpy as np
    import torch
    
    from Dataset import CardImageDataset
    from torch.utils.data import DataLoader, SequentialSampler
    
    import trainUtils as tu
    
    np.random.seed(42)
    torch.manual_seed(47)
    
    if CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print('cuDNN version: ',torch.backends.cudnn.version(), ' is available: ', torch.backends.cudnn.is_available())
        torch.backends.cudnn.enabled = True

    dataset = CardImageDataset(root_dir='../data', header_file='gicsd_labels.csv', image_dir='images')
    indexes = np.arange(dataset.limit)
#    print(indexes.shape)
    
    data_sampler = SequentialSampler(indexes)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, drop_last=True, num_workers=worker_n)
    
    model_path = '../artifacts/train_checkpoint.pt' if CUDA else '../artifacts/train_cpu.pt'
    model, _, _ = tu.init_model(path=model_path,
                                                load_model=True,
                                                cuda=CUDA)

    labels, predictions = tu.predict_from_loader(model, data_loader, CUDA)

    from sklearn import metrics
    print('Generating confusion matrix ... ')
    con_mat = metrics.confusion_matrix(labels, predictions)
    print('Confusion matrix generated.')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    print('ploting confusion matrix ... ')
    con_mat_df = pd.DataFrame(con_mat,
                              index=dataset.classes,
                              columns=dataset.classes)
    
    plt.figure('confusion matrix', figsize=(8,8))
    cmap = sns.light_palette('blue', as_cmap=True)
    out_plot = sns.heatmap(con_mat_df, annot=True, fmt='0.2f', cmap=cmap)
    out_plot.invert_yaxis()
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.savefig('../misc/confusion_matrix.png', bbox_inches='tight')
    print('Confusion matrix plotted.')

def predict(path='../data/images/GICSD_4_1_33.png', CUDA=False):
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    
    import numpy as np
    import torch
    
    from Dataset import CardImageDataset, load_image_from_path
    import trainUtils as tu
    
    np.random.seed(42)
    torch.manual_seed(47)
    
    if CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print('cuDNN version: ',torch.backends.cudnn.version(), ' is available: ', torch.backends.cudnn.is_available())
        torch.backends.cudnn.enabled = True
    
    model_path = '../artifacts/train_checkpoint.pt' if CUDA else '../artifacts/train_cpu.pt'
    model, _, _ = tu.init_model(path=model_path,
                                load_model=True,
                                cuda=CUDA)
    dataset = CardImageDataset(root_dir='../data', header_file='gicsd_labels.csv', image_dir='images')
    
    image = load_image_from_path(path)
    x = torch.unsqueeze(image, dim=0)

    if CUDA:
        x = x.cuda()
    with torch.no_grad():
        model.eval()
        out = model(x)
    out = out.cpu().detach().numpy()
    pred = np.argmax(out, axis=1)
    
    import matplotlib.pyplot as plt
    plt.figure('prediction', figsize=(5,5))
    plt.imshow(image[0]); plt.axis('off')
    plt.title('prediction:\n{}'.format(dataset.classes[pred[0]]))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    predict_all_dataset(batch_size=40,
                        worker_n=4,
                        CUDA=True)
    
    predict(path='../data/images/GICSD_4_1_33.png', CUDA=True)

