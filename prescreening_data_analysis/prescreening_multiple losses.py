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

random_data = loadmat('random.mat')
vote_data = loadmat('vote.mat')
mean_data = loadmat('mean.mat')
combine_25 = loadmat('combine_25.mat')
combine_50 = loadmat('combine_50.mat')
combine_75 = loadmat('combine_75.mat')

sns.set_palette(palette='Set2')

# def get_false_area(dict_label, threshold):
#     mask = np.zeros(len(dict_label['screening_prob_frame'][0]))
#     mask[dict_label['screening_prob_frame'][0] >= threshold] = 1
#     for i in range(len(dict_label['gt_frame'][0])):
#
#         #FP (gt=0, prediction=1)
#         if dict_label['gt_frame'][0][i] == 0 and mask[i] == 1:
#     return
# print(random_data['gt_frame'][0][0])
# print(get_false_area(random_data,0.5))

def get_FP_prescreened(dict_label, threshold):
    # make a boolean mask for specific threshold for prescreening
    mask = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask2 = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask[dict_label['screening_prob_frame'][0] >= threshold] = 1
    mask2[dict_label['pred_pixels'][0] > 0] = 1
    tn, fp, fn, tp = confusion_matrix(dict_label['gt_frame'][0], mask*mask2).ravel()
    return tn, fp, fn, tp

def get_FP_prescreened_just_seg(dict_label):
    # make a boolean mask for specific threshold for prescreening
    mask = np.zeros(len(dict_label['pred_pixels'][0]))
    mask[dict_label['pred_pixels'][0] > 0] = 1
    tn, fp, fn, tp = confusion_matrix(dict_label['gt_frame'][0], mask).ravel()
    return tn, fp, fn, tp


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


index = np.array([0,1,3,5])
columns = ['Vote', 'Random', 'Mean', 'Combine (25%)', 'Combine (50%)', 'Combine (75%)']
df_FP = pd.DataFrame(index=range(7), columns=columns)
df_FN = df_FP.copy()

for ii, data1 in enumerate([vote_data, random_data, mean_data, combine_25, combine_50, combine_75]):
    for i in range(4):
        tn, fp, fn, tp = get_FP_prescreened(data1, index[i])
        df_FP.iloc[np.int(i)][ii] = fp / (fp + tn)
        df_FN.iloc[np.int(i)][ii] = fn / (fn + tp)

#No screening
for ii, data1 in enumerate([vote_data, random_data, mean_data, combine_25, combine_50, combine_75]):
    tn, fp, fn, tp = get_FP_prescreened_just_seg(data1)
    df_FP.iloc[np.int(4)][ii] = fp / (fp + tn)
    df_FN.iloc[np.int(4)][ii] = fn / (fn + tp)

#Dice-BCE
for ii, data1 in enumerate([vote_data_dice_bce, random_data_dice_bce, mean_data_dice_bce, combine_25_dice_bce, combine_50_dice_bce, combine_75_dice_bce]):
    tn, fp, fn, tp = get_FP_prescreened_just_seg(data1)
    df_FP.iloc[np.int(5)][ii] = fp / (fp + tn)
    df_FN.iloc[np.int(5)][ii] = fn / (fn + tp)

#Weighted BCE 
for ii, data1 in enumerate([vote_data_weighted_bce, random_data_weighted_bce, mean_data_weighted_bce, combine_25_weighted_bce, combine_50_weighted_bce, combine_75_weighted_bce]):
    tn, fp, fn, tp = get_FP_prescreened_just_seg(data1)
    df_FP.iloc[np.int(6)][ii] = fp / (fp + tn)
    df_FN.iloc[np.int(6)][ii] = fn / (fn + tp)


fig, axes = plt.subplots(2, 1, figsize=(6,6))

df_FP.plot.bar(rot=0, ax=axes[0], legend=False)
#axes[0].set_xlabel("Method")
axes[0].set_ylabel("FP rate")
axes[0].set_xticklabels(labels = ['T=0', 'T=1', 'T=3', 'T=5', 'DSC', 'DSC-BCE', 'W-BCE'], minor=False)
axes[0].legend(bbox_to_anchor= (0.9, 1.5), ncol=3, title='Method')

df_FN.plot.bar(rot=0, ax=axes[1], legend=False)
axes[1].set_xlabel("Method")
axes[1].set_ylabel("FN rate")
axes[1].set_xticklabels(labels = ['T=0', 'T=1', 'T=3', 'T=5', 'DSC', 'DSC-BCE', 'W-BCE'], minor=False)
#handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc="upper center", ncol=3, title='Label sampling')
fig.tight_layout()
plt.show()