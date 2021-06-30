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
vote_data  = loadmat('vote.mat')
mean_data = loadmat('mean.mat')
combine_25 = loadmat('combine_25.mat')
combine_50 = loadmat('combine_50.mat')
combine_75 = loadmat('combine_75.mat')

#random_data = loadmat('mean.mat')


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
    preds = np.copy(dict_label['pred_pixels'][0])
    preds[preds>0] = 1
    return (mask[0]*preds).astype(bool)


#def get_diff_thresh_dice

## Dice on all frames for different thresholds
#random_data = get_dice(random_data)
mask = get_boolean_prescreened(random_data,1)
plt.figure(figsize=(6, 6))
data = np.array([])
false_negatives = np.array([])
thresholds = np.linspace(0,5,100)
leg = np.array(['random_data', 'vote_data', 'mean_data','combine_25','combine_50','combine_75'])
for ii, data1 in enumerate([random_data, vote_data, mean_data,combine_25,combine_50,combine_75]):
    data1 = get_dice(data1)
    data = np.array([])
    false_negatives = np.array([])
    for i in thresholds:
        mask = get_boolean_prescreened(data1,i)
        subset = np.copy(mask)
        for i in range(len(subset)):
            if positive_frames[i] == False & mask[i] == False:
                subset[i] = False
            else:
                subset[i] = True
        # On screened frams including false negatives
        data = np.append(data,np.mean((data1['dice'][0]*mask.astype(float))[mask]))
        # Only frames that get passed classifier
        #data = np.append(data,np.mean((random_data['dice'][0]*mask.astype(float))[mask]))
        # false negatives
        false_negatives = np.append(false_negatives,positive_frames[~mask].sum()/positive_frames.sum())
    plt.plot(thresholds,data, label = leg[ii])
plt.plot(thresholds,false_negatives)
plt.ylabel("Dice score")
plt.xlabel("Threshold values")
plt.legend() 
plt.tight_layout()

## Histogram of false positive pixels on negative frames
random_data = get_dice(random_data)
# SET THRESHOLD
thresh = 5
mask = get_boolean_prescreened(random_data,thresh)
plt.figure(figsize=(6, 6))
# Non-pre-screen
preds = np.copy(random_data['pred_pixels'][0])
preds[preds>0] = 1
data = random_data['pred_pixels'][0][negative_frames*preds.astype(bool)]

plt.hist(data, label = "no-screen")
# Get pre-screened
for i in np.linspace(0,5,11):
    mask = get_boolean_prescreened(random_data,i)
    data = (random_data['pred_pixels'][0]*mask.astype(float))[negative_frames*mask]
    plt.hist(data, label = "screen, thresh {}".format(i))
plt.xlabel("Number of FP pixels per negative segmented frame")
plt.ylabel("Number of negative frames")
plt.legend() 
plt.show()
