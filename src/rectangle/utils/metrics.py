import torch
from torch import nn

# Loss function

class DiceLoss(nn.Module):
  """ Loss function based on Dice-Sorensen Coefficient (L = 1 - Dice)
  Input arguments:
    soft : boolean, default = True
           Select whether to use soft labelling or not. If true, dice calculated
           directly on sigmoid output without converting to binary. If false,
           sigmoid output converted to binary based on threshold value
    smooth : float, default = 1e-7
             Smoothing value to add to numerator and denominator of Dice. Low
             value will prevent inf or nan occurrence.
    threshold : float, default = 0.5
                Threshold of sigmoid activation values to convert to binary.
                Only applied if soft=False.
  """
  # Standard Dice loss, with variable smoothing constant
  def __init__(self, soft=True, threshold=0.5, eps=1e-7):
      super().__init__()
      self.eps = eps
      self.soft = soft
      self.threshold = threshold

  def forward(self, inputs, targets):
    # Assume already in int form - binarise function available
    # Seems to perform very well without binary - soft dice?

    if not self.soft:
      inputs = self.BinaryDice(inputs, self.threshold)

    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()

    intersection = torch.sum(inputs * targets)
    dice = ((2. * intersection) + self.eps) / \
            (torch.sum(inputs) + torch.sum(targets) + self.eps)

    return (1 - dice)

  @staticmethod
  def BinaryDice(image, threshold=0.5):
    return (image > threshold).int()


class Precision(object):
  """ Precision metric (TP/(TP+FP))
  """
  def __init__(self, eps=1e-7):
    super().__init__()
    self.eps = eps

  def __call__(self, inputs, targets):
    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()

    TP = torch.sum(inputs * targets)
    FP = torch.sum((inputs == 1) & (targets == 0))

    return (TP + self.eps)/(TP + FP + self.eps)


class Recall(object):
  """ Recall metric (TP/(TP+FN))
  """
  def __init__(self, eps=1e-7):
    super().__init__()
    self.eps = eps

  def __call__(self, inputs, targets):
    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()

    TP = torch.sum(inputs * targets)
    FP = torch.sum((inputs == 0) & (targets == 1))

    return (TP + self.eps)/(TP + FP + self.eps)
    

class Accuracy(object):
  """ Simple binary classifier accuracy
  """
  def __init__(self):
    super().__init__()

  def __call__(self, inputs, targets):
    # inputs = inputs.view(-1).float()
    # targets = targets.view(-1).float()

    correct = (torch.round(inputs) == targets).sum().item()

    return correct / inputs.size(0)


class BCE2d(nn.Module):
  def __init__(self):
    super(BCE2d, self).__init__()
    self.criterion = nn.BCELoss(weight=None, size_average=True)  
  def forward(self, pred, label):  
    pred = pred.view(pred.size(0),-1)           
    label = label.view(label.size(0),-1)
    loss = self.criterion(pred, label)
    return loss

class WCE2d(nn.Module):
  def __init__(self):
      super().__init__()
  def forward(self, pred, label):
    img_size = label.size(2)*label.size(3)
    pred = pred.view(pred.size(0),-1).float()
    eps = 1e-6
    pred = torch.clip(pred, min=eps, max=1-eps)
    label = label.view(label.size(0),-1).float()
    weight_2 = torch.tensor(torch.sum(label, axis=1)/img_size).unsqueeze(1)
    weight_1 = 1-weight_2
    loss = -(weight_1 * (pred * torch.log(label)) + weight_2 * ((1 - pred) * torch.log(1 - label)))
    return torch.mean(loss)


# class WCE2d(nn.Module):
#   def __init__(self):
#     super().__init__()
#   def forward(self, pred, label):  
#     pred = pred.view(pred.size(0),-1)           
#     label = label.view(label.size(0),-1)
#     pred = torch.sum(pred, axis=1)
#     label = torch.sum(label, axis=1)
#     img_size = label.size(2) * label.size(3)
#     weights =  (img_size-torch.sum(label, axis=1))/(torch.sum(label, axis=1)+1e-6)
#     # weights = (label.size(1)-torch.sum(label, axis=0))/(torch.sum(label, axis=0)+1e-6)
#     print('label',label.shape, 'pred', pred.shape, 'weights',weights.shape)
#     criterion = nn.BCEWithLogitsLoss(weight=None, size_average=True, pos_weight=weights)  
#     loss = criterion(pred.float(), label.float())
#     return loss
