from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# Fields of the loaded .mat files
# gt_frame = framewise binary ground truth
# screening_prob_frame = framewise probability predicted
# gt_pixels = pixelwise #pixels ground truth
# pred_pixels = pixelwise #pixels predicted
# tp_gt_pred_pixels = pixelwise #pixels true positive


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
            dict_label['pred_pixels'] + dict_label['gt_pixels'] + 1e-7)
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

names = ['dice', 'weighted-bce', 'dice-bce']
for idx, data in enumerate([combine_25, combine_25_weighted_bce, combine_25_dice_bce]):
    print(f'Dice for segmentation : {names[idx]}')
    mean = np.mean(get_dice(data)['dice'][0][predicted_negative(data)])
    median = (np.median(get_dice(data)['dice'][0][predicted_negative(data)]))
    std = (np.std(get_dice(data)['dice'][0][predicted_negative(data)]))
    print(f'Mean: {mean}, Median: {median}, std: {std}')

print('Chicken')
data = combine_25
for threshold in [0,1,3,5]:
    dice_all = get_dice(data)
    print(f'Prescreened threshold = {threshold}')
    mean = (np.mean(dice_all['dice'][0][predicted_negative_prescreened(data,threshold)]))
    median = (np.median(dice_all['dice'][0][predicted_negative_prescreened(data,threshold)]))
    std = (np.std(dice_all['dice'][0][predicted_negative_prescreened(data,threshold)]))
    print(f'Mean: {mean}, Median: {median}, std: {std}')

print('Bubble tea')
