from torch.nn.modules import loss
from .discrepancy import mmd_loss, mk_mmd_loss
from functools import partial
from torch import zeros, ones, FloatTensor, cat, long 
from torch.nn import NLLLoss
from numpy.random import choice
from numpy import arange

class DA:
    def __init__(self, active, da_type, da_layer, n_layer, layer_offset, n_kernel, da_factor, classifier):
        self.active = bool(active)
        self.da_type = da_type
        self.da_layer = da_layer
        self.n_layer = n_layer
        self.n_kernel = n_kernel
        self.da_factor = da_factor
        self.layer_offset = layer_offset
        self.classifier = classifier
        
        if self.active:
            if self.da_type == 'DDC':
                self.da_loss_func = mmd_loss
            elif self.da_type == 'DAN':
                self.da_loss_func = partial(mk_mmd_loss, num=self.n_kernel)
            elif self.da_type == 'DANN':
                self.da_loss_func = NLLLoss()
            else:
                raise NotImplementedError()
        
    def sample(self, S_x, T_x):
        if self.active:
            batch = min(S_x.shape[0], T_x.shape[0])
            S_idx = choice(arange(S_x.shape[0]), batch)
            T_idx = choice(arange(T_x.shape[0]), batch)
            return FloatTensor(S_x[S_idx]), FloatTensor(T_x[T_idx]), zeros(batch), ones(batch)
        return zeros(([0]+list(S_x.shape[1:]))), zeros(([0]+list(S_x.shape[1:]))), zeros((0)), ones((0))

    def loss(self, S_feature : list, T_feature : list):
        if self.active:
            assert len(S_feature) == len(T_feature), "# of features should be the same, got {} for source and {} for target".format(len(S_feature), len(T_feature))
            
            if self.da_type == 'DANN':
                prediction = self.classifier(cat((S_feature[self.layer_offset], T_feature[self.layer_offset]), dim=0), self.da_factor)
                target = cat((zeros(S_feature[0].shape[0]), ones(T_feature[0].shape[0])), dim=0).to(long).to(prediction.device)
                return self.da_factor * self.da_loss_func(prediction, target)
            else:
                _loss = 0
                for i in range(self.layer_offset, self.layer_offset+self.n_layer):
                    _loss = _loss + self.da_loss_func(S_feature[i], T_feature[i])
                return _loss * self.da_factor
        return zeros(1, device=S_feature[0].device)
