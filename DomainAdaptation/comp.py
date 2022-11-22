import torch
from .can import CAN

class DA:
    def __init__(self, active, da_type, da_layer, n_layer, layer_offset, n_kernel, kernel_mul, da_factor, dataloader_src, dataloader_tgt, thre, st_thre, eps, intra_only):
        self.active = bool(active)
        self.da_type = da_type
        self.da_layer = da_layer
        self.n_layer = n_layer
        self.n_kernel = n_kernel
        self.kernel_mul = kernel_mul
        self.da_factor = da_factor
        self.layer_offset = layer_offset
        self.thre = thre
        self.eps = eps
        self.intra_only = intra_only
        
        if self.active:
            if self.da_type == 'CAN':
                self.can = CAN(dataloader_src, dataloader_tgt, n_layer, layer_offset, n_kernel, kernel_mul, thre, st_thre, eps, intra_only, num_classes=2)
            else:
                raise NotImplementedError()
        
    def sample(self, feature_extractor, regressioner):
        assert self.active, 'Should be active to sample from DomainAdaptation'

        dynamic_data_src, target_data_src, dynamic_data_tgt, target_data_tgt, mask_st = self.can.update_data(feature_extractor, regressioner)
        
        # split the data into pos and neg
        pos_idx_src = (target_data_src == 1).nonzero()[:, 0]
        neg_idx_src = (target_data_src == 0).nonzero()[:, 0]
        pos_idx_tgt = (target_data_tgt == 1).nonzero()[:, 0]
        neg_idx_tgt = (target_data_tgt == 0).nonzero()[:, 0]

        return dynamic_data_src[pos_idx_src], target_data_src[pos_idx_src], dynamic_data_src[neg_idx_src], target_data_src[neg_idx_src], dynamic_data_tgt[pos_idx_tgt], target_data_tgt[pos_idx_tgt], dynamic_data_tgt[neg_idx_tgt], target_data_tgt[neg_idx_tgt], mask_st

    def loss(self, S_feature : list, T_feature : list, S_label : torch.tensor = None, T_label : torch.tensor = None):
        if self.active:
            assert len(S_feature) == len(T_feature), "# of features should be the same, got {} for source and {} for target".format(len(S_feature), len(T_feature))
            
            if self.da_type == 'CAN':
                loss = self.can.calc_loss(S_feature[self.layer_offset:self.layer_offset+self.n_layer], S_label, T_feature[self.layer_offset:self.layer_offset+self.n_layer], T_label)
                return self.da_factor * loss['cdd']
            else:
                raise NotImplementedError()
            
        return torch.zeros(1, device=S_feature[0].device)
