import rectangle as rect
import h5py
from rectangle import model
import torch
import numpy as np
import os 
from rectangle.model.networks import DenseNet as DenseNet 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms
from scipy.io import savemat

class ResNet(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.network = model

  def forward(self, x):
    x = self.network(x)
    return x


####### SET THRESHOLDS
segmentation_threshold = 0.5 

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####### CHANGE PATHS
# Ensemble path
num_ensemble = 3
""" latest_model = ['52.pth', '44.pth', '91.pth']
path_str = './dataset/af1_models/af1_pt5_combine_25/model/'
output_name = 'combine_25'
latest_model = ['52.pth', '48.pth', '115.pth']
path_str = './dataset/af1_models/af1_pt5_combine_50/model/'
output_name = 'combine_50'
latest_model = ['52.pth', '27.pth', '97.pth']
path_str = './dataset/af1_models/af1_pt5_combine_75/model/'
output_name = 'combine_75'
latest_model = ['47.pth', '76.pth', '88.pth']
path_str = './dataset/af1_models/af1_pt5_mean/model/'
output_name = 'mean'
latest_model = ['67.pth', '94.pth', '54.pth']
path_str = './dataset/af1_models/af1_pt5_random/model/'
output_name = 'random' """
latest_model = ['66.pth', '78.pth', '45.pth']
path_str = './dataset/af1_models/af1_pt5_vote/model/'
output_name = 'vote'

# Classifier path
class_model = torch.load("./dataset/classifiers/resnet_affine2", map_location = torch.device(device))

# Test data
test_file = h5py.File('./dataset/valtest_joint.h5', 'r')

### Utils
def standardise(image):
    means = image.mean(dim=(1,2,3), keepdim=True)
    stds = image.std(dim=(1,2,3), keepdim=True)
    return (image - means.expand(image.size())) / stds.expand(image.size())

# Loading ensemble segmentation
model_paths = [os.path.join(path_str, 'model_'+ str(idx), latest_model[idx]) for idx in range(num_ensemble)]
seg_models = [rect.model.networks.UNet(n_layers=5, device=device, gate=None) for e in range(int(num_ensemble))]
for n, m in enumerate(seg_models):
    m.load_state_dict(torch.load(model_paths[n], map_location= device))

# Loading the test data
test_DS = rect.utils.io.H5DataLoader(test_file)
test_DL = DataLoader(test_DS, batch_size = 24, shuffle = False)

# Put models in evaluation mode
class_model.eval()
for seg_model in seg_models:
    seg_model.eval()

# Initialise arrays for gathering data

## FRAME-WISE
screening_prob_frame = np.array([])
gt_frame = np.array([])

## PIXEL-WISE
#tp_pixels = segmap*gt_label
gt_pixels = np.array([])
pred_pixels = np.array([])
tp_gt_pred_pixels = np.array([])

normalise_img = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.Normalize(0.449, 0.226)])


# Remove the gradients
with torch.no_grad():
    # Inference of all the test data
    for jj, (images_test, labels_test) in enumerate(test_DL):
        # Put into GPU if available
        if use_cuda: 
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
        
        # Obtain positive and negative frames 
        gt_frames = [(1 in label) for label in labels_test]
        # Obtain prediction for classifier 
        class_preds = class_model(normalise_img(images_test)) 
        # Normalise images for segmentation network 
        norm_images_test = standardise(images_test)
        combined_predictions = torch.zeros_like(labels_test, dtype = float)
        # Obtain segmentation predictions 
        for seg_model in seg_models:
            seg_predictions = (seg_model(norm_images_test) > segmentation_threshold).clone().detach()
            combined_predictions += seg_predictions
        # All segmentation results - only on positive frames 
        combined_predictions = (combined_predictions >= 2) #Majority vote 
        ### Pre-screened results only ### 
        gt_frame = np.append(gt_frame, np.array(gt_frames))
        screening_prob_frame = np.append(screening_prob_frame,class_preds.detach().cpu().numpy())

        pred_pixels = np.append(pred_pixels,torch.sum(combined_predictions, dim = [1,2,3]).detach().cpu().numpy())
        gt_pixels = np.append(gt_pixels,torch.sum(labels_test, dim = [1,2,3]).detach().cpu().numpy())
        tp_gt_pred_pixels = np.append(tp_gt_pred_pixels,torch.sum(labels_test*combined_predictions, dim = [1,2,3]).detach().cpu().numpy())

prescreen_dict = {'label':output_name,'gt_frame':gt_frame,'screening_prob_frame':screening_prob_frame,'pred_pixels':pred_pixels,\
    'gt_pixels':gt_pixels,'tp_gt_pred_pixels':tp_gt_pred_pixels}
savemat(output_name+'.mat', prescreen_dict)

