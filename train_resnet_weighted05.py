import os
import random
from numpy.lib.shape_base import split
import requests
import h5py
import torch
from torchvision import models
from torchvision import transforms
from torch import nn
from torch.optim import Adam
from os import path
from datetime import date
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import RandomAffine

class Affine(object):
  """ Affine augmentation of image. Wrapper for torchvision RandomAffine.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    prob : int, default = 0.3
           Probability of augmentation occuring at each pass.
    degrees : int, default = 5
              Range of possible rotation (-degrees, +degrees). Set to None for
              no rotation.
    translate : float, default = 0.1
                Range of possible translation. Set to None for no translation.
    scale : tuple, default = (0.9, 1.1)
            Range of possible scaling. Set to None for no scaling.
    shear : int, default = 5
            Range of possible shear rotation (-shear, +shear). Set to None for
            no shear.
  """
  def __init__(self, prob=0.3,\
    degrees=5, translate=0.1,
    scale=(0.9,1.1), shear=5):
    super().__init__() 
    self.prob = prob
    self.degrees = degrees
    self.translate = translate
    self.scale = scale
    self.shear = shear

  def __call__(self, image):
    rand_ = random.uniform(0,1)
    if rand_ < self.prob:
      RandAffine_ = RandomAffine(degrees=self.degrees, translate=(self.translate,self.translate),
                                scale=self.scale, shear=self.shear)
      image = RandAffine_(image)
    return image

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
    #self.num_subjects = len(self.split_keys)
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

class ResNet(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    """     self.normalise = transforms.Normalize(0.449, 0.226)
    self.resample = torch.nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) """
    self.normalise = transforms.Compose([
                    transforms.Resize(224),
                    transforms.Normalize(0.449, 0.226)])
    self.network = model

  def forward(self, x):
    x = self.normalise(x)
    x = self.network(x)
    return x

def MakeResNet(freeze_weights=True, pretrain=False):
  cnn = models.resnext101_32x8d(pretrained=pretrain)
  cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  cnn.fc = torch.nn.Linear(in_features=2048, out_features=1, bias=True)
  if freeze_weights:
    for param in cnn.parameters():
      param.requires_grad=False
  return ResNet(cnn)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_available = torch.cuda.is_available()

ResNet_model = MakeResNet(freeze_weights= False, pretrain= True)

if cuda_available:
    ResNet_model.cuda()

#train_data = h5py.File('/Users/iani/Documents/Segmentation_project/in4it/dataset/train.h5', 'r')
train_data = h5py.File('/raid/candi/Iani/MRes_project/dataset/train.h5', 'r')
train_all = ClassifyDataLoader(train_data)

#Split up training data into 0.7:0.1 split 
train_len = np.int(np.round((0.7/0.8)*train_all.__len__()))
val_len = train_all.__len__() - train_len

train_dataset = torch.utils.data.Subset(train_all, list(np.arange(0,train_len)))
val_dataset = torch.utils.data.Subset(train_all, list(np.arange(train_len,train_len + val_len)))

train_DL = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_DL = torch.utils.data.DataLoader(val_dataset, batch_size = 8, shuffle = True)

def accuracy_score_nosig(predicted, target): 
  """
  Accuracy score for model with final layer of sigmoid so no need to apply again 
  """
  #Apply sigmoid 
  correct = (torch.round(predicted) == target.cpu()).sum().item()

  return correct / predicted.size(0)

def accuracy_score(predicted, target): 

  """
  Accuracy for classifier model with no final sigmoid layer 
  """

  predicted = torch.sigmoid(predicted)
  predicted = predicted > 0.5
  correct = (predicted == target).sum().item()

  return correct / predicted.size(0)

no_epochs = 200
avg_loss = np.zeros(no_epochs,)
avg_loss_val = np.zeros(no_epochs,)
avg_accuracy_val = np.zeros(no_epochs,)

best_loss = np.inf

#Saving file names
loss_log_file = 'classification_loss_resnet_weighted05.csv'
saved_model_file = 'resnet_weighted05' #Name of saved moel 

### TRAINING AND VALIDATION FOR N NO OF EPOCHS ###
with open(loss_log_file, 'w') as loss:
    loss.write('''\
      Epoch, train_loss, val_loss, val_accuracy
      ''')

optimiser = torch.optim.Adam(ResNet_model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight = torch.tensor(0.5))#, pos_weight = torch.tensor(0.25))
Apply_transform = Affine(prob = 0.3, degrees = 5, translate= 0.1, scale = (0.9,1.1), shear = 5) 

for epoch in range(no_epochs):

    #Restart loss vals for every epoch #
    all_loss_train = [] 
    all_loss_val = [] 
    all_accuracy = [] 

    ### TRAINING ####
    ResNet_model.train()
    for idx, (image, label) in enumerate(train_DL):

        if cuda_available:
            image, label = image.cuda(), label.cuda()
        
        #Apply data augmentation 
        image = Apply_transform(image)

        #Update weights of NN
        optimiser.zero_grad()
        output = ResNet_model(image)
        loss = loss_fn(output, label.float())
        loss.backward()
        optimiser.step()
            
        all_loss_train.append(loss.item())
    
    #Save average epoch loss 
    avg_loss[epoch] = np.mean(all_loss_train)
    print('Epoch %d, Average loss: %.3f' % (epoch, avg_loss[epoch]))

    ### VALIDATION ###
    ResNet_model.eval()
    for idx, (image, label) in enumerate(val_DL):
        if cuda_available:
            image, label = image.cuda(), label.cuda()

        with torch.no_grad():
            output = ResNet_model(image)
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
        torch.save(ResNet_model, saved_model_file)

    ### SAVING LOSS VALUES
    with open(loss_log_file, 'a') as loss:
        all_vals =  np.concatenate([np.array(epoch).reshape(1,), avg_loss[epoch].reshape(1,), avg_loss_val[epoch].reshape(1,), avg_accuracy_val[epoch].reshape(1,)], axis = 0)
        np.savetxt(loss, np.reshape(all_vals, [1,-1]), '%s', delimiter =",")