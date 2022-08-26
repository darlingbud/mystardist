import torch
import torch.nn.functional as F

class L1_BCELoss(torch.nn.Module):
    
    def __init__(self,scale=[1,1]):
        super(L1_BCELoss, self).__init__()
        assert len(scale)==2
        self.scale = scale
        
    def forward(self, prediction, target_dists,prob):
        #Predicted distances errors are weighted by object prob
        l1loss = F.l1_loss(prediction[0],target_dists,size_average=True, reduce=False)
        #weights = self.getWeights(target_dists
        l1loss = torch.mean(prob*l1loss)
        bceloss = F.binary_cross_entropy(prediction[1],prob, weight=None, size_average=True, reduce=True)
        return self.scale[0]*l1loss + self.scale[1]*bceloss,l1loss,bceloss
    