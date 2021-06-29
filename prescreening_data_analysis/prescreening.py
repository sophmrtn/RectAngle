from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
# Fields of the loaded .mat files
# gt_frame = framewise binary ground truth
# screening_prob_frame = framewise probability predicted
# gt_pixels = pixelwise #pixels ground truth
# pred_pixels = pixelwise #pixels predicted
# tp_gt_pred_pixels = pixelwise #pixels true positive

random_data = loadmat('random.mat')
""" vote_data = loadmat('vote')
mean_data = loadmat('mean')
combine_data = loadmat('combine')
overlap_data = loadmat('overlap') """

positive_frames = random_data['gt_frame'].astype(bool)[0]
negative_frames = np.abs(random_data['gt_frame']-1).astype(bool)[0]


def get_dice(dict_label):
    # get dice for all the frames
    dict_label['dice'] = 2*dict_label['tp_gt_pred_pixels']/(dict_label['pred_pixels']+dict_label['gt_pixels'] + 1e-8)
    return dict_label

def get_boolean_prescreened(dict_label,threshold):
    # make a boolean mask for specific threshold for prescreening
    mask = np.copy(dict_label['screening_prob_frame'])
    mask[mask<threshold] = False
    mask[mask>=threshold] = True
    return mask[0].astype(bool)

## Dice on all frames for different thresholds
random_data = get_dice(random_data)
mask = get_boolean_prescreened(random_data,1)
plt.figure(figsize=(3, 3))
data = np.array([])
thresholds = np.array([0,0.25,0.5,0.75,1.0,1.25])
for i in thresholds:
    mask = get_boolean_prescreened(random_data,i)
    data = np.append(data,np.mean(random_data['dice'][0][mask]))
plt.plot(thresholds,data)
plt.tight_layout()

## Histogram of false positive pixels on negative frames
random_data = get_dice(random_data)
mask = get_boolean_prescreened(random_data,10)
plt.figure(figsize=(3, 3))
# Non-pre-screen
data = random_data['pred_pixels'][0][negative_frames]
plt.hist(data, label = "no-screen", bins = 10)
# Get pre-screened
data = (random_data['pred_pixels'][0]*mask.astype(float))[negative_frames]
plt.hist(data, label = "screen", bins = 10)
plt.xlabel("Number of FP pixels per negative segmented frame")
plt.ylabel("Number of negative frames")
plt.legend() 
plt.show()
