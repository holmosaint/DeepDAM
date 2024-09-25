import torch
from torch import zeros
from .can import CAN

class DA:
    def __init__(self, active, da_type, da_layer, n_layer, layer_offset, n_kernel, kernel_mul, da_factor, dataloader_src, dataloader_tgt, thre, st_thre, eps, intra_only, writer):
        """
        Initializes the Domain Adaptation component.
        Parameters:
            active (bool): Flag to activate domain adaptation.
            da_type (str): Type of domain adaptation.
            da_layer (str): Domain adaptation layer.
            n_layer (int): Number of layers.
            layer_offset (int): Offset for the layers.
            n_kernel (int): Number of kernels.
            kernel_mul (float): Kernel multiplier.
            da_factor (float): Domain adaptation factor.
            classifier (object): Classifier object.
            dataloader_src (object): Source data loader.
            dataloader_tgt (object): Target data loader.
            thre (float): Threshold value.
            st_thre (float): Second threshold value.
            eps (float): Epsilon value.
            intra_only (bool): Flag for intra-domain adaptation only.
            writer (object): Writer object for logging.
        Raises:
            AssertionError: If `active` is True and `da_type` is not 'CAN'.
        """
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
            assert self.da_type == 'CAN'
            self.can = CAN(dataloader_src, dataloader_tgt, n_layer, layer_offset, n_kernel, kernel_mul, thre, st_thre, eps, intra_only, num_classes=2, writer=writer)
        
    def sample(self, feature_extractor, classifier):
        """
        Samples and splits the data into positive and negative indices for both source and target domains.
        Args:
            feature_extractor: A model or function used to extract features from the data.
            classifier: A model or function used to perform classification on the data.
        Returns:
            tuple: A tuple containing the following elements:
                - dynamic_data_src_pos (Tensor): Dynamic source data with positive indices.
                - target_data_src_pos (Tensor): Target source data with positive indices.
                - dynamic_data_src_neg (Tensor): Dynamic source data with negative indices.
                - target_data_src_neg (Tensor): Target source data with negative indices.
                - dynamic_data_tgt_pos (Tensor): Dynamic target data with positive indices.
                - pred_target_data_tgt_pos (Tensor): Predicted target data with positive indices.
                - dynamic_data_tgt_neg (Tensor): Dynamic target data with negative indices.
                - pred_target_data_tgt_neg (Tensor): Predicted target data with negative indices.
                - mask_st (Tensor): Mask used for the data.
        """

        dynamic_data_src, target_data_src, dynamic_data_tgt, pred_target_data_tgt, mask_st = self.can.update_data(feature_extractor, classifier)
        
        # split the data into pos and neg
        pos_idx_src = (target_data_src == 1).nonzero()[:, 0]
        neg_idx_src = (target_data_src == 0).nonzero()[:, 0]
        pos_idx_tgt = (pred_target_data_tgt == 1).nonzero()[:, 0]
        neg_idx_tgt = (pred_target_data_tgt == 0).nonzero()[:, 0]

        return dynamic_data_src[pos_idx_src], target_data_src[pos_idx_src], dynamic_data_src[neg_idx_src], target_data_src[neg_idx_src], dynamic_data_tgt[pos_idx_tgt], pred_target_data_tgt[pos_idx_tgt], dynamic_data_tgt[neg_idx_tgt], pred_target_data_tgt[neg_idx_tgt], mask_st

    def loss(self, S_feature : list, T_feature : list, S_label : torch.tensor = None, T_label : torch.tensor = None):
        """
        Computes the domain adaptation loss between source and target features.
        Args:
            S_feature (list): List of source domain features.
            T_feature (list): List of target domain features.
            S_label (torch.tensor, optional): Tensor of source domain labels. Defaults to None.
            T_label (torch.tensor, optional): Tensor of target domain labels. Defaults to None.
        Returns:
            torch.tensor: The computed loss value if domain adaptation is active; otherwise, a tensor of zeros.
        Raises:
            AssertionError: If the number of source and target features do not match.
            AssertionError: If the domain adaptation type is not 'CAN'.
        """
        if self.active:
            assert len(S_feature) == len(T_feature), "# of features should be the same, got {} for source and {} for target".format(len(S_feature), len(T_feature))
            
            assert self.da_type == 'CAN'
            loss = self.can.calc_loss(S_feature[self.layer_offset:self.layer_offset+self.n_layer], S_label, T_feature[self.layer_offset:self.layer_offset+self.n_layer], T_label)
            return self.da_factor * loss['cdd']
        
        return zeros(1, device=S_feature[0].device)
