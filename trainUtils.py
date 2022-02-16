#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time

import torch
import torch.optim as optim
from Model import CardModel

from sklearn import metrics


def init_model(path='F:\my proj\lincode\modified_mobilenet_v2_features_state_dict', load_model=False, cuda=False, lr=1e-3, decay_points=[], decay=0.1):    
    print('initizalizing Model...')
    model = CardModel(path=path, load=(not load_model))
    if load_model:
        model.load_state_dict(torch.load(path), strict=True)
    if cuda:
        model.cuda()
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9,0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_points, gamma=decay)
    print('Model loaded...')
    return model, optimizer, scheduler


def calculate_predictions(out):
    predicted = np.argmax(out, axis=1)
    return predicted

def train(model, optimizer, loader, epoch_i, cuda=False):
    model.train()
    loss = 0.0
    avg = 0.0
    start = time.time()
    loss_hist = []
    predictions = []
    labels = []
    for batch_idx, data in enumerate(loader, 1):
        batch_time_s = time.time()

        y = torch.squeeze(data['label'])
        x = data['image']
        if cuda:
            x = x.cuda()
            y = y.cuda()
        
        optimizer.zero_grad()
        out = model(x)
        loss = model.criter(out, y)
        loss.backward()
        optimizer.step()
        
        out = out.cpu().detach().numpy()
        preds = calculate_predictions(out)
        predictions += list(preds)
        y = y.cpu().detach().numpy()
        labels += list(y)
        # metrics
        accuracy = metrics.accuracy_score(y, preds)
        precision = metrics.precision_score(y, preds, average='weighted',zero_division=1)
        recall = metrics.recall_score(y, preds, average='weighted',zero_division=1)
        f1_score = metrics.f1_score(y, preds, average='weighted',zero_division=1)
        
        loss = float(loss.detach())
        loss_hist.append(loss)
        avg += loss
        
        spent_time = time.time() - batch_time_s
        out_str = '\rTRAIN Epoch: {} Loss: {:.6f} Acc: {:5.2f} preci: {:.3f} recall: {:.3f} f1 score: {:.3f} time: {:.2f}{}'.format(
                epoch_i, loss, 100*accuracy, precision, recall, f1_score, spent_time, 10*' ')
        print('\r'+out_str, end='')
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average='weighted',zero_division=1)
    recall = metrics.recall_score(labels, predictions, average='weighted',zero_division=1)
    f1_score = metrics.f1_score(labels, predictions, average='weighted',zero_division=1)
    
    total_time = time.time() - start
    avg /= len(loader)
    out_str = 'TRAIN Epoch: {} Avg Loss: {:.6f} Acc: {:5.2f} preci: {:.3f} recall: {:.3f} f1 score: {:.3f} time: {:.2f}{}'.format(
            epoch_i, avg, 100*accuracy, precision, recall, f1_score, total_time, 10*' ')
    print('\r'+out_str)
    
    return loss_hist, avg, accuracy, precision, recall, f1_score


def validation(model, loader, epoch_i, cuda=False, type_t='VAL'):
    with torch.no_grad():
        model.eval()
        loss = 0.0
        avg = 0.0
        start = time.time()
        loss_hist = []
        predictions = []
        labels = []
        for batch_idx, data in enumerate(loader, 1):
            batch_time_s = time.time()

            y = torch.squeeze(data['label'])
            x = data['image']
            if cuda:
                x = x.cuda()
                y = y.cuda()
            
            out = model(x)
            loss = float(model.criter(out, y).detach_())
            loss_hist.append(loss)
            avg += loss
            
            out = out.cpu().detach().numpy()
            preds = calculate_predictions(out)
            predictions += list(preds)
            y = y.cpu().detach().numpy()
            labels += list(y)
            # metrics
            accuracy = metrics.accuracy_score(y, preds)
            precision = metrics.precision_score(y, preds, average='weighted',zero_division=1)
            recall = metrics.recall_score(y, preds, average='weighted',zero_division=1)
            f1_score = metrics.f1_score(y, preds, average='weighted',zero_division=1)
        
            spent_time = time.time() - batch_time_s
            
            out_str = '\r{} Epoch: {} Loss: {:.6f} Acc: {:5.2f} preci: {:.3f} recall: {:.3f} f1 score: {:.3f} time: {:.2f}{}'.format(
                    type_t, epoch_i, loss, 100*accuracy, precision, recall, f1_score, spent_time, 10*' ')
            print('\r'+out_str, end='')
        
        labels = np.array(labels)
        predictions = np.array(predictions)
        accuracy = metrics.accuracy_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions, average='weighted',zero_division=1)
        recall = metrics.recall_score(labels, predictions, average='weighted',zero_division=1)
        f1_score = metrics.f1_score(labels, predictions, average='weighted',zero_division=1)
        
        total_time = time.time() - start
        avg /= len(loader)
        out_str = '{} Epoch: {} Avg Loss: {:.6f} Acc: {:5.2f} preci: {:.3f} recall: {:.3f} f1 score: {:.3f} time: {:.2f}{}'.format(
                type_t, epoch_i, avg, 100*accuracy, precision, recall, f1_score, total_time, 10*' ')
        print('\r'+out_str)
        
        return loss_hist, avg, accuracy, precision, recall, f1_score
    
    
def predict_from_loader(model, loader, cuda=False):
    with torch.no_grad():
        model.eval()
        loss = 0.0
        avg = 0.0
        start = time.time()
        loss_hist = []
        predictions = []
        labels = []
        for batch_idx, data in enumerate(loader, 1):
            batch_time_s = time.time()

            y = torch.squeeze(data['label'])
            x = data['image']
            if cuda:
                x = x.cuda()
                y = y.cuda()
            
            out = model(x)
            loss = float(model.criter(out, y).detach_())
            loss_hist.append(loss)
            avg += loss
            
            out = out.cpu().detach().numpy()
            preds = calculate_predictions(out)
            predictions += list(preds)
            y = y.cpu().detach().numpy()
            labels += list(y)
            # metrics
            accuracy = metrics.accuracy_score(y, preds)
            precision = metrics.precision_score(y, preds, average='weighted',zero_division=1)
            recall = metrics.recall_score(y, preds, average='weighted',zero_division=1)
            f1_score = metrics.f1_score(y, preds, average='weighted',zero_division=1)
        
            spent_time = time.time() - batch_time_s
            
            out_str = '\r{:5.2f}% {}/{} Loss: {:.6f} Acc: {:5.2f} preci: {:.3f} recall: {:.3f} f1 score: {:.3f} time: {:.2f}{}'.format(
                    100*batch_idx/len(loader), batch_idx, len(loader), loss, 100*accuracy, precision, recall, f1_score, spent_time, 10*' ')
            print('\r'+out_str, end='')
            
        labels = np.array(labels)
        predictions = np.array(predictions)
        accuracy = metrics.accuracy_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions, average='weighted',zero_division=1)
        recall = metrics.recall_score(labels, predictions, average='weighted',zero_division=1)
        f1_score = metrics.f1_score(labels, predictions, average='weighted',zero_division=1)
        
        total_time = time.time() - start
        avg /= len(loader)
        out_str = 'Avg Loss: {:.6f} Acc: {:5.2f} preci: {:.3f} recall: {:.3f} f1 score: {:.3f} time: {:.2f}{}'.format(
                avg, 100*accuracy, precision, recall, f1_score, total_time, 10*' ')
        print('\r'+out_str)
        
        return labels, predictions

