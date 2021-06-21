import os
import random
from numpy.lib.shape_base import split
import requests

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torchvision import transforms, models
from torchvision.transforms import Normalize
#from data_loader import ClassifyDataLoader 

import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from os import path, makedirs
from datetime import date
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from scipy.ndimage import laplace
import tensorflow as tf
class ClassifyDataLoader(torch.utils.data.Dataset):

  def __init__(self, file, keys=None):
    """ Dataloader for hdf5 files, with labels converted to classifier labels
    Input arguments:
      file : h5py File object
             Loaded using h5py.File(path : string)
      keys : list, default = None
             Keys from h5py file to use. Useful for train-val-test split.
             If None, keys generated from entire file.
    """
    
    super().__init__()

    self.file = file
    if not keys:
      keys = list(file.keys())
    
    self.split_keys = [key.split('_') for key in keys]
    start_subj = int(self.split_keys[0][1])
    last_subj = int(self.split_keys[-1][1])
    self.num_subjects = (last_subj - start_subj)+ 1  #Add 1 to account for 0 idx python
    self.subjects = [key[1] for key in self.split_keys if key[0] == 'frame']
    #self.subjects = np.linspace(start_subj, last_subj, 
    #                            self.num_subjects+1, dtype=int)

  def __len__(self):
        return self.num_subjects
  
  def __getitem__(self, index):
    
    subj_ix = self.subjects[index]
    image_key = 'frame_' + subj_ix
    image = torch.unsqueeze(torch.tensor(self.file[image_key][()].astype('float32')), dim=0)

    label_batch = torch.cat([torch.unsqueeze(torch.tensor(
        self.file[f'label_{subj_ix}_0{label_ix}' ]
        [()].astype('float32')), dim=0) for label_ix in range(3)])
    
    label_vote = torch.sum(label_batch, dim=(1,2))
    sum_vote = torch.sum(label_vote != 0)
    
    #print(sum_vote)
    if sum_vote >= 2:
      label = torch.tensor([1.0])
    else:
      label = torch.tensor([0.0])

    return(image, label)

class DenseNet(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.normalise = Normalize(0.449, 0.226)
    self.resample = torch.nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.features = torch.nn.Sequential(*list(model.features)[1:])
    self.flatten = torch.nn.Flatten()
    self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=291456, out_features=1, bias=True), torch.nn.Sigmoid())

  def forward(self, x):
    x = self.normalise(x)
    x = self.resample(x)
    x = self.features(x)
    #x = torch.squeeze(x)
    x = self.classifier(x)
    return x

def MakeDenseNet(freeze_weights=True, pretrain=True):
  
  cnn = models.densenet161(pretrained=pretrain)

  if freeze_weights:
    for param in cnn.parameters():
      param.requires_grad=False

  return DenseNet(cnn)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_available = torch.cuda.is_available()

DenseNet_model = MakeDenseNet(freeze_weights= False, pretrain= True)

if cuda_available:
    DenseNet_model.cuda()

#train_data = h5py.File('/Users/iani/Documents/Segmentation_project/in4it/dataset/train.h5', 'r')
train_data = h5py.File('/raid/candi/Iani/MRes_project/dataset/dataset_zip/train.h5', 'r')
train_dataset = ClassifyDataLoader(train_data)
train_DL = torch.utils.data.DataLoader(train_dataset, batch_size = 8)

val_data = h5py.File('/raid/candi/Iani/MRes_project/dataset/dataset_zip/val.h5', 'r')
val_dataset = ClassifyDataLoader(val_data)
val_DL = torch.utils.data.DataLoader(val_dataset, batch_size = 8)

def accuracy_score(predicted, target): 

    correct = (torch.round(predicted) == target.cpu()).sum().item()

    return correct / predicted.size(0)

no_epochs = 200
avg_loss = np.zeros(no_epochs,)
avg_loss_val = np.zeros(no_epochs,)
avg_accuracy_val = np.zeros(no_epochs,)

best_loss = np.inf

### TRAINING AND VALIDATION FOR N NO OF EPOCHS ###
with open('classification_loss_file2.csv', 'w') as loss:
    loss.write('''\
      Epoch, train_loss, val_loss, val_accuracy
      ''')

optimiser = torch.optim.Adam(DenseNet_model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight = torch.tensor(0.25))

for epoch in range(no_epochs):

    #Restart loss vals for every epoch #
    all_loss_train = [] 
    all_loss_val = [] 
    all_accuracy = [] 

    ### TRAINING ####
    DenseNet_model.train()
    for idx, (image, label) in enumerate(train_DL):

        if cuda_available:
            image, label = image.cuda(), label.cuda()
        
        #Update weights of NN
        optimiser.zero_grad()
        output = DenseNet_model(image)
        loss = loss_fn(output, label.float())
        loss.backward()
        optimiser.step()
            
        all_loss_train.append(loss.item())
    
    #Save average epoch loss 
    avg_loss[epoch] = np.mean(all_loss_train)
    print('Epoch %d, Average loss: %.3f' % (epoch, avg_loss[epoch]))

    ### VALIDATION ###
    DenseNet_model.eval()
    for idx, (image, label) in enumerate(val_DL):
        if cuda_available:
            image, label = image.cuda(), label.cuda()

        with torch.no_grad():
            output = DenseNet_model(image)
            loss = loss_fn(output,label.float())
            accuracy = accuracy_score(output.cpu(), label.cpu())
                
        all_loss_val.append(loss.item())
        all_accuracy.append(accuracy)

    avg_loss_val[epoch] = np.mean(all_loss_val)  
    avg_accuracy_val[epoch] = np.mean(all_accuracy)

    print('[Epoch %d, Average val Loss: %.3f, Accuracy: %.3f' % (epoch, avg_loss_val[epoch], avg_accuracy_val[epoch]))

    #Saving model if loss is lower than best loss 
    if avg_loss_val[epoch] < best_loss: 
        print('New best model')
        best_loss = avg_loss_val[epoch]
        torch.save(DenseNet_model, 'classification_model_weighted_loss')

    ### SAVING LOSS VALUES
    with open('classification_loss_file2.csv', 'a') as loss:
        all_vals =  np.concatenate([np.array(epoch).reshape(1,), avg_loss[epoch].reshape(1,), avg_loss_val[epoch].reshape(1,), avg_accuracy_val[epoch].reshape(1,)], axis = 0)
        np.savetxt(loss, np.reshape(all_vals, [1,-1]), '%s', delimiter =",")