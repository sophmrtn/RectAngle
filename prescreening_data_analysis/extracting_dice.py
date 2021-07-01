from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

# Fields of the loaded .mat files
# gt_frame = framewise binary ground truth
# screening_prob_frame = framewise probability predicted
# gt_pixels = pixelwise #pixels ground truth
# pred_pixels = pixelwise #pixels predicted
# tp_gt_pred_pixels = pixelwise #pixels true positive

sns.set_palette(palette='Set2')

random_data = loadmat('random.mat')
vote_data = loadmat('vote.mat')
mean_data = loadmat('mean.mat')
combine_25 = loadmat('combine_25.mat')
combine_50 = loadmat('combine_50.mat')
combine_75 = loadmat('combine_75.mat')

random_data_dice_bce = loadmat('random_dice_bce.mat')
vote_data_dice_bce = loadmat('vote_dice_bce.mat')
mean_data_dice_bce = loadmat('mean_dice_bce.mat')
combine_25_dice_bce = loadmat('combine_25_dice_bce.mat')
combine_50_dice_bce = loadmat('combine_50_dice_bce.mat')
combine_75_dice_bce = loadmat('combine_75_dice_bce.mat')

random_data_weighted_bce = loadmat('random_weighted_bce.mat')
vote_data_weighted_bce = loadmat('vote_weighted_bce.mat')
mean_data_weighted_bce = loadmat('mean_weighted_bce.mat')
combine_25_weighted_bce = loadmat('combine_25_weighted_bce.mat')
combine_50_weighted_bce = loadmat('combine_50_weighted_bce.mat')
combine_75_weighted_bce = loadmat('combine_75_weighted_bce.mat')

def get_dice(dict_label):
    # get dice for all the frames
    dict_label['dice'] = 2 * dict_label['tp_gt_pred_pixels'] / (
            dict_label['pred_pixels'] + dict_label['gt_pixels'] + 1e-8)
    return dict_label

def predicted_negative_prescreened(dict_label, threshold):
    # make a boolean mask for specific threshold for prescreening
    mask = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask[dict_label['screening_prob_frame'][0] >= threshold] = 1
    mask2 = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask2[dict_label['pred_pixels'][0] > 0] = 1
    return (mask*mask2).astype(bool)

def predicted_negative(dict_label):
    # make a boolean mask for specific threshold for prescreening
    mask2 = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask2[dict_label['pred_pixels'][0] > 0] = 1
    return mask2.astype(bool)

# Pre
data = random_data
print('Not prescreened')
print(np.mean(get_dice(data)['dice'][0][predicted_negative(data)]))
print(np.median(get_dice(data)['dice'][0][predicted_negative(data)]))
print(np.std(get_dice(data)['dice'][0][predicted_negative(data)]))
print('Prescreened t=0')
theshold = 0
print(np.mean(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.median(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.std(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print('Prescreened t=1')
theshold = 1
print(np.mean(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.median(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.std(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print('Prescreened t=3')
theshold = 3
print(np.mean(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.median(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.std(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print('Prescreened t=5')
theshold = 5
print(np.mean(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.median(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
print(np.std(get_dice(data)['dice'][0][predicted_negative_prescreened(data,theshold)]))
