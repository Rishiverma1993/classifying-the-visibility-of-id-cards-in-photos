#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def training_process(batch_size=32,
                     n_epochs=100,
                     lr=1e-3,
                     decay=0.1,
                     decay_points=[50],
                     patience=5,
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

    test_indexes = np.load('../artifacts/test_indexes.npy')
    val_indexes = np.load('../artifacts/val_indexes.npy')
    train_indexes = np.load('../artifacts/train_indexes.npy')
    np.random.shuffle(test_indexes)
    np.random.shuffle(val_indexes)
    np.random.shuffle(train_indexes)
    
    dataset = CardImageDataset(root_dir='../data', header_file='gicsd_labels.csv', image_dir='images')
    
    test_sampler = SequentialSampler(test_indexes)
    val_sampler = SequentialSampler(val_indexes)
    train_sampler = SequentialSampler(train_indexes)
    
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=True, num_workers=worker_n)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True, num_workers=worker_n)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=worker_n)
    
    model, optimizer, scheduler = tu.init_model(path='../artifacts/modified_mobilenet_v2_features_state_dict.pth',
                                                load_model=False,
                                                cuda=CUDA,
                                                lr=lr,
                                                decay_points=decay_points,
                                                decay=decay)

    count = 0
    best_loss = 999999.9
    epoch_val = 0
    best_f1 = 0.0
    try:
        loss_hist, avg, accuracy, precision, recall, f1_score = tu.validation(model, val_loader, -1, cuda=CUDA, type_t='VAL')
        for epoch in range(1, n_epochs+1):
            loss_hist, avg, accuracy, precision, recall, f1_score = tu.train(model, optimizer, train_loader, epoch, cuda=CUDA)
            val_loss_hist, val_avg, val_accuracy, val_precision, val_recall, val_f1_score = tu.validation(model, val_loader, epoch, cuda=CUDA, type_t='VAL')
            scheduler.step()
            if best_loss > val_avg or val_f1_score > best_f1:
                best_loss = val_avg
                best_f1 = val_f1_score
                epoch_val = epoch
                torch.save(model.state_dict(), '../artifacts/train_checkpoint.pt')
                count = 0
            else:
                count += 1
            if count > patience:
                break
    except KeyboardInterrupt:
        print('\n','-' * 89)
        print('Exiting from training early')
        
    #del val_list
    #model.empty_cache(gc)
    print('\nReached best validation: {:.6f} f1 score'.format(best_loss, best_f1), ' ... at: ', epoch_val)
    model.load_state_dict(torch.load('../artifacts/train_checkpoint.pt'), strict=True)
    val_loss_hist, val_avg, val_accuracy, val_precision, val_recall, val_f1_score = tu.validation(model, test_loader, epoch_val, cuda=CUDA, type_t='TEST')
            
    model.cpu()
    torch.save(model.state_dict(), '../artifacts/train_cpu.pt')    

if __name__ == '__main__':
    training_process(batch_size=32,
                     n_epochs=100,
                     lr=1e-3,
                     decay=0.1,
                     decay_points=[50],
                     patience=5,
                     worker_n=4,
                     CUDA=True)

