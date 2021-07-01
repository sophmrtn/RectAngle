from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys

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


def get_FP_prescreened(dict_label, threshold):
    # make a boolean mask for specific threshold for prescreening
    mask = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask[dict_label['screening_prob_frame'][0] >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(dict_label['gt_frame'][0], mask).ravel()
    return tn, fp, fn, tp


index = np.linspace(0, 5, 6)
columns = ['random', 'vote', 'mean', 'combine_25', 'combine_50', 'combine_75']
df_FP = pd.DataFrame(index=index, columns=columns)
df_FN = df_FP.copy()

for ii, data1 in enumerate([random_data, vote_data, mean_data, combine_25, combine_50, combine_75]):
    for i in index:
        tn, fp, fn, tp = get_FP_prescreened(data1, i)
        df_FP.iloc[np.int(i)][ii] = fp / (fp + tn)
        df_FN.iloc[np.int(i)][ii] = fn / (fn + tp)

print(df_FP)
print(df_FN)

fig, axes = plt.subplots(2, 1, figsize=(6,6))

df_FP.plot.bar(rot=0, ax=axes[0], legend=False)
axes[0].set_xlabel("Threshold values")
axes[0].set_ylabel("FP rate")
axes[0].legend(bbox_to_anchor= (0.08, 1.01), ncol=3, title='Label sampling')
df_FN.plot.bar(rot=0, ax=axes[1], legend=False)
axes[1].set_xlabel("Threshold values")
axes[1].set_ylabel("FN rate")
#handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc="upper center", ncol=3, title='Label sampling')
fig.tight_layout()

df_FP.plot.bar(rot=0)
plt.xlabel("Threshold values")
plt.ylabel("FP rate")
plt.legend(title="Label sampling", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
df_FN.plot.bar(rot=0)
plt.xlabel("Threshold values")
plt.ylabel("FN rate")
plt.legend(title="Label sampling", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# random_data = loadmat('mean.mat')

positive_frames = random_data['gt_frame'].astype(bool)[0]
negative_frames = np.abs(random_data['gt_frame'] - 1).astype(bool)[0]


def get_dice(dict_label):
    # get dice for all the frames
    dict_label['dice'] = 2 * dict_label['tp_gt_pred_pixels'] / (
            dict_label['pred_pixels'] + dict_label['gt_pixels'] + 1e-8)
    return dict_label


def get_boolean_prescreened(dict_label, threshold):
    # make a boolean mask for specific threshold for prescreening
    mask = np.zeros(len(dict_label['screening_prob_frame'][0]))
    mask[dict_label['screening_prob_frame'][0] >= threshold] = 1
    preds = np.copy(dict_label['pred_pixels'][0])
    preds[preds > 0] = 1
    return (mask * preds).astype(bool)


# def get_diff_thresh_dice

## Dice on all frames for different thresholds
# random_data = get_dice(random_data)
mask = get_boolean_prescreened(random_data, 1)
plt.figure(figsize=(6, 6))
data = np.array([])
false_negatives = np.array([])
thresholds = np.linspace(0, 5, 100)
leg = np.array(['random', 'vote', 'mean', 'combine_25', 'combine_50', 'combine_75'])

for ii, data1 in enumerate([random_data, vote_data, mean_data, combine_25, combine_50, combine_75]):
    data1 = get_dice(data1)
    data = np.array([])
    false_negatives = np.array([])
    for i in thresholds:
        mask = get_boolean_prescreened(data1, i)
        subset = np.copy(mask)
        for i in range(len(subset)):
            if positive_frames[i] == False & mask[i] == False:
                subset[i] = False
            else:
                subset[i] = True
        # On screened frames including false negatives
        data = np.append(data, np.mean((data1['dice'][0] * mask.astype(float))[mask]))
        # Only frames that get passed classifier
        # data = np.append(data,np.mean((random_data['dice'][0]*mask.astype(float))[mask]))
        # false negatives
        false_negatives = np.append(false_negatives, positive_frames[~mask].sum() / positive_frames.sum())
    plt.plot(thresholds, data, label=leg[ii])
# plt.plot(thresholds,false_negatives)
plt.ylabel("Dice score")
plt.xlabel("Threshold values")
plt.legend(title="Label sampling")
plt.tight_layout()

## Histogram of false positive pixels on negative frames
combine_25 = get_dice(combine_25)
# SET THRESHOLD
thresh = 5
mask = get_boolean_prescreened(combine_25, thresh)
plt.figure(figsize=(6, 6))
# Non-pre-screen
preds = np.copy(combine_25['pred_pixels'][0])
preds[preds > 0] = 1
data = combine_25['pred_pixels'][0][negative_frames * preds.astype(bool)]

bin_ranges = np.linspace(0, 62356, 11)  # 64189 is the maximal FP pixel count for combine_25
plt.hist(data, bins=bin_ranges, edgecolor='black', label="no screen")

# Get pre-screened
color_list = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
# 'tab:gray', 'tab:olive', 'tab:cyan', 'navy', 'bisque']
for i in np.linspace(0, 5, 6):
    mask = get_boolean_prescreened(combine_25, i)
    data = (combine_25['pred_pixels'][0] * mask.astype(float))[negative_frames * mask]
    plt.hist(data, bins=bin_ranges, color=color_list[np.int(i)], edgecolor='black',
             label="thresh {}".format(i))  # range=[0, sys.float_info.max]
plt.xlabel("Number of FP pixels per negative segmented frame")
plt.ylabel("Number of negative frames")
plt.legend(title="Screening strategy")
plt.show()
