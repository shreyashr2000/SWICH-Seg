import torch.nn as nn
class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.eps=1e-7
    def forward(self, x, target):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims =  tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        if torch.sum(intersection) + torch.sum(cardinality) == 0:
            return 0.0
        if torch.sum(intersection)==0:
            return 1
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)
