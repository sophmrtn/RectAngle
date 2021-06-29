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
vote_data = loadmat('vote.mat')
mean_data = loadmat('mean.mat')
combine_50_data = loadmat('combine_50.mat')
combine_75_data = loadmat('combine_75.mat')

positive_frames = random_data['gt_frame'].astype(bool)[0]
negative_frames = np.abs(random_data['gt_frame']-1).astype(bool)[0]


def get_dice(dict_label):
    # get dice for all the frames
    dict_label['dice'] = 2*dict_label['tp_gt_pred_pixels']/(dict_label['pred_pixels']+dict_label['gt_pixels'] + 1e-8)
    return dict_label

def get_boolean_prescreened(dict_label,threshold):
    # make a boolean mask for specific threshold for prescreening
    mask = np.copy(dict_label['screening_prob_frame'])
    mask[mask<threshold] = 0.
    mask[mask>=threshold] = 1.
    preds = dict_label['pred_pixels'][0]
    preds[preds>0] = 1
    return (mask[0]*preds).astype(bool)

#def get_diff_thresh_dice

## Dice on all frames for different thresholds
random_data = get_dice(random_data)
mask = get_boolean_prescreened(random_data,1)
plt.figure(figsize=(3, 3))
data = np.array([])
false_negatives = np.array([])
thresholds = np.linspace(0,20,100)
for i in thresholds:
    mask = get_boolean_prescreened(random_data,i)
    # On all frames
    #data = np.append(data,np.mean((random_data['dice'][0]*mask.astype(float))))
    # Only on screened frames
    data = np.append(data,np.mean((random_data['dice'][0]*mask.astype(float))[mask]))
    # false negatives
    false_negatives = np.append(false_negatives,positive_frames[~mask].sum()/positive_frames.sum())
print(false_negatives)
plt.plot(thresholds,data)
plt.plot(thresholds,false_negatives)
plt.tight_layout()

## Histogram of false positive pixels on negative frames
random_data = get_dice(random_data)
# SET THRESHOLD
thresh = 0.5
mask = get_boolean_prescreened(random_data,thresh)
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
