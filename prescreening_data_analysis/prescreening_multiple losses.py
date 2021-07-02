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


def get_FP_prescreened(dict_label, threshold):
    # make a boolean mask for specific threshold for prescreening
    mask = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask[dict_label['screening_prob_frame'][0] >= threshold] = 1
    mask2 = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask2[dict_label['pred_pixels'][0] > 0] = 1
    tn, fp, fn, tp = confusion_matrix(dict_label['gt_frame'][0], mask * mask2).ravel()
    return tn, fp, fn, tp


def get_FP_prescreened_just_seg(dict_label):
    # make a boolean mask for specific threshold for prescreening
    mask = np.zeros(len(dict_label['pred_pixels'][0]))
    mask[dict_label['pred_pixels'][0] > 0] = 1
    tn, fp, fn, tp = confusion_matrix(dict_label['gt_frame'][0], mask).ravel()
    return tn, fp, fn, tp


def get_false_area(dict_label, threshold):
    mask = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask2 = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask[dict_label['screening_prob_frame'][0] >= threshold] = 1
    mask2[dict_label['pred_pixels'][0] > 0] = 1
    FP_pixels = []
    FN_pixels = []
    mask3 = mask*mask2
    # loop through indices
    for k in range(len(dict_label['gt_frame'][0])):

        # FP (gt=0, prediction=1)
        if dict_label['gt_frame'][0][k] == 0 and mask3[k] == 1:
            # get number of predicted pixels
            FP_pixels.append(dict_label['pred_pixels'][0][k])

        # FN (gt=1, prediction=0)
        if dict_label['gt_frame'][0][k] == 1 and mask3[k] == 0:
            # get number of ground truth pixels
            FN_pixels.append(dict_label['gt_pixels'][0][k])

    pixel_area = 0.177994000000000 * 0.161290000000000  # mm^2
    mean_FP_area = 0
    mean_FN_area = 0
    if len(FP_pixels) != 0:
        mean_FP_area = np.mean(FP_pixels) * pixel_area
    if len(FN_pixels) != 0:
        mean_FN_area = np.mean(FN_pixels) * pixel_area
    print(mean_FP_area)

    return mean_FP_area, mean_FN_area


def get_false_area_just_seg(dict_label):
    mask = np.zeros(len(dict_label['pred_pixels'][0]))
    mask[dict_label['pred_pixels'][0] > 0] = 1
    FP_pixels = []
    FN_pixels = []

    # loop through indices
    for k in range(len(dict_label['gt_frame'][0])):

        # FP (gt=0, prediction=1)
        if dict_label['gt_frame'][0][k] == 0 and mask[k] == 1:
            # get number of predicted pixels
            FP_pixels.append(dict_label['pred_pixels'][0][k])

        # FN (gt=1, prediction=0)
        if dict_label['gt_frame'][0][k] == 1 and mask[k] == 0:
            # get number of ground truth pixels
            FN_pixels.append(dict_label['gt_pixels'][0][k])

    pixel_area = 0.177994000000000 * 0.161290000000000  # mm^2
    mean_FP_area = 0
    mean_FN_area = 0
    if len(FP_pixels) != 0:
        mean_FP_area = np.mean(FP_pixels) * pixel_area
    if len(FN_pixels) != 0:
        mean_FN_area = np.mean(FN_pixels) * pixel_area

    return mean_FP_area, mean_FN_area


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

index = np.array([0, 1, 3, 5])
columns = ['Vote', 'Random', 'Mean', 'Combine (25%)', 'Combine (50%)', 'Combine (75%)']
df_FP = pd.DataFrame(index=range(7), columns=columns)
df_FN = df_FP.copy()
df_FP_area = df_FP.copy()
df_FN_area = df_FP.copy()

for ii, data1 in enumerate([vote_data, random_data, mean_data, combine_25, combine_50, combine_75]):
    for i in range(4):
        tn, fp, fn, tp = get_FP_prescreened(data1, index[i])
        df_FP.iloc[int(i)][ii] = fp / (fp + tn)
        df_FN.iloc[int(i)][ii] = fn / (fn + tp)
        area_FP, area_FN = get_false_area(data1, index[i])
        df_FP_area.iloc[int(i)][ii] = area_FP
        df_FN_area.iloc[int(i)][ii] = area_FN

# No screening
for ii, data1 in enumerate([vote_data, random_data, mean_data, combine_25, combine_50, combine_75]):
    tn, fp, fn, tp = get_FP_prescreened_just_seg(data1)
    df_FP.iloc[int(4)][ii] = fp / (fp + tn)
    df_FN.iloc[int(4)][ii] = fn / (fn + tp)
    area_FP, area_FN = get_false_area_just_seg(data1)
    df_FP_area.iloc[int(4)][ii] = area_FP
    df_FN_area.iloc[int(4)][ii] = area_FN

# Dice-BCE
for ii, data1 in enumerate(
        [vote_data_dice_bce, random_data_dice_bce, mean_data_dice_bce, combine_25_dice_bce, combine_50_dice_bce,
         combine_75_dice_bce]):
    tn, fp, fn, tp = get_FP_prescreened_just_seg(data1)
    df_FP.iloc[int(5)][ii] = fp / (fp + tn)
    df_FN.iloc[int(5)][ii] = fn / (fn + tp)
    area_FP, area_FN = get_false_area_just_seg(data1)
    df_FP_area.iloc[int(5)][ii] = area_FP
    df_FN_area.iloc[int(5)][ii] = area_FN

# Weighted BCE
for ii, data1 in enumerate(
        [vote_data_weighted_bce, random_data_weighted_bce, mean_data_weighted_bce, combine_25_weighted_bce,
         combine_50_weighted_bce, combine_75_weighted_bce]):
    tn, fp, fn, tp = get_FP_prescreened_just_seg(data1)
    df_FP.iloc[int(6)][ii] = fp / (fp + tn)
    df_FN.iloc[int(6)][ii] = fn / (fn + tp)
    area_FP, area_FN = get_false_area_just_seg(data1)
    df_FP_area.iloc[int(6)][ii] = area_FP
    df_FN_area.iloc[int(6)][ii] = area_FN


# fig, axes = plt.subplots(2, 1)
#
# df_FP.plot.bar(rot=0, ax=axes[0], legend=False)
# # axes[0].set_xlabel("Method")
# axes[0].set_ylabel("FP rate")
# axes[0].set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DSC', 'DSC-BCE', 'W-BCE'], minor=False)
# axes[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
#                mode="expand", borderaxespad=0.)
#
# df_FN.plot.bar(rot=0, ax=axes[1], legend=False)
# #axes[1].set_xlabel("Method")
# axes[1].set_ylabel("FN rate")
# axes[1].set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DSC', 'DSC-BCE', 'W-BCE'], minor=False)
# # handles, labels = axes[0].get_legend_handles_labels()
# # fig.legend(handles, labels, loc="upper center", ncol=3, title='Label sampling')
# fig.tight_layout()
# plt.show()

# fig, axes = plt.subplots(2, 1)
#
# df_FP_area.plot.bar(rot=0, ax=axes[0], legend=False)
# # axes[0].set_xlabel("Method")
# axes[0].set_ylabel("FP area ($mm^2$)")
# axes[0].set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DSC', 'DSC-BCE', 'W-BCE'], minor=False)
# axes[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
#                mode="expand", borderaxespad=0.)
# df_FN_area.plot.bar(rot=0, ax=axes[1], legend=False)
# #axes[1].set_xlabel("Method")
# axes[1].set_ylabel("FN area ($mm^2$)")
# axes[1].set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DSC', 'DSC-BCE', 'W-BCE'], minor=False)
# # handles, labels = axes[0].get_legend_handles_labels()
# # fig.legend(handles, labels, loc="upper center", ncol=3, title='Label sampling')
# fig.tight_layout()
# plt.show()

fig, axes = plt.subplots(2, 2)

axs = axes[0,0]
df_FP.plot.bar(ax=axs, legend=False)
axs.axvline(x=3.5, ymin=0, ymax=1, linestyle='--', color="k")
# axes[0].set_xlabel("Method")
axs.set_ylabel("FP rate")
# axs.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
axs.set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DICE', 'DICE-BCE', 'W-BCE'], minor=False)
# axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
#                mode="expand", borderaxespad=0.)

axs = axes[1,0]
df_FN.plot.bar(ax=axs, legend=False)
axs.axvline(x=3.5, ymin=0, ymax=1, linestyle='--', color="k")
#axes[1].set_xlabel("Method")
axs.set_ylabel("FN rate")
axs.set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DICE', 'DICE-BCE', 'W-BCE'], minor=False)
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=3, title='Label sampling')


axs = axes[0,1]

df_FP_area.plot.bar(ax=axs, legend=False)
axs.axvline(x=3.5, ymin=0, ymax=1, linestyle='--', color="k")
# axes[0].set_xlabel("Method")
axs.set_ylabel("FP area ($mm^2$)")
# axs.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
axs.set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DICE', 'DICE-BCE', 'W-BCE'], minor=False)
#axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
#               mode="expand", borderaxespad=0.)

axs = axes[1,1]
df_FN_area.plot.bar(ax=axs, legend=False)
axs.axvline(x=3.5, ymin=0, ymax=1, linestyle='--', color="k")
#axes[1].set_xlabel("Method")
axs.set_ylabel("FN area ($mm^2$)")
axs.set_xticklabels(labels=['T=0', 'T=1', 'T=3', 'T=5', 'DICE', 'DICE-BCE', 'W-BCE'], minor=False)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,bbox_to_anchor=(0.04, 1.02, 1., .102), borderaxespad=0.)
#legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
#               mode="expand", borderaxespad=0.)
fig.tight_layout()
plt.savefig("output.png", bbox_inches="tight")
plt.show()