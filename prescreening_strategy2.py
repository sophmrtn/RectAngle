import rectangle as rect
import h5py
from rectangle import model
import torch
import numpy as np
import os 
from rectangle.model.networks import DenseNet as DenseNet 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

####### SET THRESHOLDS
segmentation_threshold = 0.5 
classification_thresholds = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8])

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####### CHANGE PATHS
# Ensemble path
num_ensemble = 5 
latest_model = ['13.pth', '4.pth', '30.pth', '28.pth', '28.pth'] #Checked manually 
path_str = '.\dataset\ensemble'
# Classifier path
class_model = torch.load("./dataset/classifiers/dense_aug", map_location = torch.device(device))
# Test data
test_file = h5py.File('./dataset/test.h5', 'r')



### Utils
def standardise(image):
    means = image.mean(dim=(1,2,3), keepdim=True)
    stds = image.std(dim=(1,2,3), keepdim=True)
    return (image - means.expand(image.size())) / stds.expand(image.size())

def dice_score2(y_pred, y_true, eps=1e-8):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    #y_pred[y_pred < 0.5] = 0.
    #y_pred[y_pred > 0] = 1.
    
    #Calculate the number of incorrectly labelled pixels 
    numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2
    denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps
    return torch.mean(numerator / denominator)

def dice_fp(y_pred, y_true, pos_frames, neg_frames): 
    """ A function that computes dice score on positive frames, 
    and FP pixels on negative frames, based off Yipeng's metrics
    """
    dice_ = dice_score2(y_pred[pos_frames, :, :], y_true[pos_frames, :, :])
    fp = torch.sum(y_pred[neg_frames, :, :], dim = [1,2,3])

    return dice_, fp 
    


# Loading ensemble segmentation
model_paths = [os.path.join(path_str, 'model_'+ str(idx), latest_model[idx]) for idx in range(num_ensemble)]
seg_models = [rect.model.networks.UNet(n_layers=5, device=device, gate=None) for e in range(int(num_ensemble))]
for n, m in enumerate(seg_models):
    m.load_state_dict(torch.load(model_paths[n], map_location= device))

# Loading the test data
test_DS = rect.utils.io.H5DataLoader(test_file)
test_DL = DataLoader(test_DS, batch_size = 10, shuffle = False)

# Put models in evaluation mode
class_model.eval()
for seg_model in seg_models:
    seg_model.eval()

# Initialise arrays for gathering data
all_dice_screen = []
all_dice_noscreen = []

all_fp_screen = []
all_fp_noscreen = [] 

# Run for different classifications
for classification_threshold in classification_thresholds:
    # Remove the gradients
    with torch.no_grad():
        # Inference of all the test data
        for jj, (images_test, labels_test) in enumerate(test_DL):
            # Put into GPU if available
            if use_cuda: 
                images_test, labels_test = images_test.cuda(), labels_test.cuda()
            
            # Obtain positive and negative frames 
            positive_frames = [(1 in label) for label in labels_test]
            negative_frames = [not negs for negs in positive_frames]

            # Obtain prediction for classifier 
            class_preds = class_model(images_test) #If using densenet 
            
            # Normalise images for segmentation network 
            norm_images_test = standardise(images_test)

            combined_predictions = torch.zeros_like(labels_test, dtype = float)
            majority = int(np.round(len(seg_models)/2) + 1) #Majority vote number eg (num_ensembles / 2) + 1

            # Obtain segmentation predictions 
            for seg_model in seg_models:
                seg_predictions = (seg_model(norm_images_test) > segmentation_threshold).clone().detach()
                combined_predictions += seg_predictions

            # All segmentation results - only on positive frames 
            combined_predictions = (combined_predictions >= majority) #Majority vote 
            dice_noscreen, fp_noscreen = dice_fp(combined_predictions, labels_test, positive_frames, negative_frames)
            all_dice_noscreen.append(dice_noscreen)
            all_fp_noscreen.append(fp_noscreen)

            ### Pre-screened results only ### 
            prostate_idx = np.where(class_preds.cpu() > classification_threshold)[0]
            
            positive_frames_screened = [positive_frames[i] for i in prostate_idx]
            negative_frames_screened = [positive_frames[i] for i in prostate_idx]

            dice_screen, fp_screen = dice_fp(combined_predictions[prostate_idx, :,:], labels_test[prostate_idx, :,:], positive_frames_screened, negative_frames_screened)
            all_dice_screen.append(dice_screen)
            all_fp_screen.append(fp_screen)

            print(f"Dice scores: Not-screened : {dice_noscreen.detach().cpu().numpy()} | Screened : {dice_screen.detach().cpu().numpy()}")
            print(f"FP scores: Not-screened : {fp_noscreen.detach().cpu().numpy()} | Screened : {fp_screen.detach().cpu().numpy()}")


    #Obtaining plots of the histogram 

    #Obtain all unique FP scores for screen, no screen method 
    unique_fp_screen = [np.unique(fp_vals) for fp_vals in all_fp_screen if len(fp_vals) > 0]
    unique_fp_screen = np.concatenate(unique_fp_screen, axis = 0)

    #Obtain all unique FP scores for screen, no screen method 
    unique_fp_noscreen = [np.unique(fp_vals) for fp_vals in all_fp_noscreen if len(fp_vals) > 0]
    unique_fp_noscreen = np.concatenate(unique_fp_noscreen, axis = 0)

    
    plt.hist(unique_fp_noscreen, label = "noscreen")
    plt.hist(unique_fp_screen, label = "screen")
    plt.xlabel("Number of FP pixels per negative segmented frame")
    plt.legend() 
    plt.show()

