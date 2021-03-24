import torch
from torch import nn
import random
from scipy.ndimage import measurements, gaussian_filter
from torchvision.transforms import RandomAffine
import numpy as np


# Pre- and post- processing:
#   * Pre-processing:
#     - Z-score normalisation
#   * Augmentation:
#     - Affine transformation
#     - Noise (speckle and gauss)
#     - Smoothing (gauss only, savitzky-golay in TODO)
#     - Flip (L-R only)
#   * Post-processing:
#     - Binarise (variable threshold)
#     - Keep-largest-connected-component


def z_score(image):
  """ Z-Score normalisation of image. Applied image-wise (not batch-wise) such
  that each image has a mean of 0 and standard deviation of 1.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
  """
  batch_ = image.shape[0]
  for batch_iter_ in range(batch_):
    image[batch_iter_,...] = (image[batch_iter_,...] - \
                              torch.mean(image[batch_iter_,...]) / \
                              torch.std(image[batch_iter_,...]))
  return image


def Affine(image, prob=0.7,
           degrees=10, translate=0.1,
           scale=(0.9,1.1), shear=10):
  """ Affine augmentation of image. Wrapper for torchvision RandomAffine.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    prob : int, default = 0.7
           Probability of augmentation occuring at each pass.
    degrees : int, default = 10
              Range of possible rotation (-degrees, +degrees). Set to None for
              no rotation.
    translate : float, default = 0.1
                Range of possible translation. Set to None for no translation.
    scale : tuple, default = (0.9, 1.1)
            Range of possible scaling. Set to None for no scaling.
    shear : int, default = 10
            Range of possible shear rotation (-shear, +shear). Set to None for
            no shear.
  """
  rand_ = random.uniform(0,1)
  if rand_ < prob:
    RandAffine_ = RandomAffine(degrees=degrees, translate=(translate,translate),
                               scale=scale, shear=shear)
    image = RandAffine_(image)
  return image



def SpeckleNoise(image, type='speckle', mean=0, sigma=1, prob=0.7):
  """ Noise augmentation of image. Distribution scaled to image so 
  normalisation not essential.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    type : string, default = 'speckle'
           Type of noise to apply.
           Options:
                  * 'speckle': Multiplicative speckle noise
                  * 'gauss' : Additive gaussian noise
    mean : int, default = 0
           Mean of distribution for random sampling
    sigma : int, default = 1
            Standard deviation of distribution for random sampling
    prob : int, default = 0.7
           Probability of augmentation occuring at each pass.
  """
  rand_ = random.uniform(0,1)
  if rand_ < prob:
    max = torch.amax(image, dim=(1,2,3))
    noise = torch.randn(image.shape)

    for i, max_ in enumerate(max):
      mean_ = max_ * mean
      sigma_ = max_ * sigma
      noise[i,...] = mean_ + (sigma_**0.5) * noise[i,...]

    if 'speckle':
      image = noise * image
    elif 'gauss':
      image = noise + image
    else:
      raise ValueError('Invalid noise type - \
      please enter either "speckle" or "gauss".')
  return image
  

def Smooth(image, sigma=1, prob=0.7):
  """ Spatial smoothing of image. Currently only Gaussian smoothing.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    sigma : int, default = 1
            Standard deviation of smoothing kernel
    prob : int, default = 0.7
           Probability of augmentation occuring at each pass.
  """
  # Ideally add Savitzky-Golay filter instead of Gauss
  rand_ = random.uniform(0,1)
  if rand_ < prob:
    max = torch.amax(image, dim=(1,2,3))
    noise = torch.randn(image.shape)

    for i, max_ in enumerate(max):
      sigma_ = max_ * sigma_
      image_ = torch.squeeze(image[i,0,...])
      image_ = image_.detach().cpu().numpy()
      image_smooth_ = gaussian_filter(image_, sigma_)
      image[i,0,...] = torch.tensor(image_smooth_)
  return image


def Flip(image, prob=0.7):
  """ Randomly flip image in vertical axis (left and right)
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    prob : int, default = 0.7
           Probability of augmentation occuring at each pass.
  """
  rand_ = random.uniform(0,1)
  if rand_ < prob:
    image = torch.fliplr(image)
  return image


def KeepLargestComponent(image):
  """ Remove all regions of label except largest connected component.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
  """
  # FIX: Currently sets whole image to zero...
  image_batch_ = torch.squeeze(image, dim=1)
  image_batch_ = image_batch_.detach().cpu().numpy()
  batch_size_ = image_batch_.shape[0]

  for image_ in range(batch_size_):
    image_ = np.squeeze(image_)
    comp_, feat_ = measurements.label(image_)
    largest_ = (image_ == feat_).astype(int)
    image[image_,...] = torch.tensor(largest_)
  
  return image


def Binary(image, threshold=0.5):
  """ Convert float tensor to binary based on threshold value.
  Input arguments:
    image : Torch Tensor [B,C,H,W], dtype = int
    threshold : float, default = 0.5
                Value above which all intensities are converted to 1, and
                below which all intensities are converted to 0.
  """
  return (image > threshold).int()
