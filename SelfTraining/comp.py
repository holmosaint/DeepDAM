import numpy as np
import torch

class ST:
    def __init__(self, active, st_factor, exp_dynamic_data, exp_target_data, thre):
        """
        Initializes the class with the given parameters.

        Args:
            active (bool): Indicates whether the instance is active.
            st_factor (float): Self-training factor.
            exp_dynamic_data (Any): Experimental dynamic data.
            exp_target_data (Any): Experimental target data.
            thre (float): Threshold value.

        Attributes:
            active (bool): Indicates whether the instance is active.
            st_factor (float): Self-training factor.
            exp_dynamic_data (Any): Experimental dynamic data.
            exp_target_data (Any): Experimental target data.
            thre (float): Threshold value.
            exp_dynamic_data_train (Any): Training data for experimental dynamic data, initialized to None.
            exp_target_data_train (Any): Training data for experimental target data, initialized to None.
            train_available (bool): Indicates whether training data is available, initialized to False.
        """
        self.active = active
        self.st_factor = st_factor

        self.exp_dynamic_data = exp_dynamic_data
        self.exp_target_data = exp_target_data
        self.thre = thre

        self.exp_dynamic_data_train, self.exp_target_data_train = None, None
        self.train_available = False

    def sample(self, batch_size):
        """
        Samples a batch of data for training.
        Parameters:
        batch_size (int): The number of samples to retrieve.
        Returns:
        tuple: A tuple containing:
            - sample_dynamic (torch.Tensor): The dynamic data samples.
            - sample_param (torch.Tensor): The corresponding target parameters.
            - sample_domain (torch.Tensor): The domain labels for the samples.
            - sample_weight (torch.Tensor): The weights for the samples.
        If the object is not active, returns tensors filled with zeros and ones as placeholders.
        """
        batch_size = min(batch_size, self.exp_dynamic_data.shape[0])
        if self.active:

            dynamic_sample_pos, target_sample_pos, dynamic_sample_neg, target_sample_neg = self.sample_conn(int(batch_size / 2))
            sample_dynamic = torch.cat([dynamic_sample_pos, dynamic_sample_neg], dim=0)
            sample_param = torch.cat([target_sample_pos, target_sample_neg], dim=0)
            idx = np.arange(sample_dynamic.shape[0])
            np.random.shuffle(idx)
            sample_dynamic = sample_dynamic[idx]#
            sample_param = sample_param[idx].reshape(-1, 1)

            return sample_dynamic, sample_param
        
        return torch.zeros(0, self.exp_dynamic_data.shape[-1]), torch.zeros([0]), torch.ones(0), torch.FloatTensor([1])
    
    def sample_conn(self, batch_size):
        """
        Samples a batch of positive and negative connections from the training data.
        Parameters:
        -----------
        batch_size : int
            The number of samples to retrieve.
        Returns:
        --------
        tuple
            A tuple containing:
            - exp_dynamic_pos (torch.FloatTensor): A tensor of positive samples.
            - torch.ones (torch.Tensor): A tensor of ones with the same length as exp_dynamic_pos.
            - exp_dynamic_neg (torch.FloatTensor): A tensor of negative samples.
            - torch.zeros (torch.Tensor): A tensor of zeros with the same length as exp_dynamic_neg.
        Notes:
        ------
        - If the object is not active or training data is not available, returns tensors of zeros.
        - The method ensures that the batch size does not exceed the available number of samples.
        - Positive and negative indices are shuffled before sampling.
        """
        if self.active and self.train_available:
            batch_size = min(batch_size, self.exp_dynamic_data_train.shape[0])
            pos_idx = np.argwhere(self.exp_target_data_train == 1)[:, 0]
            neg_idx = np.argwhere(self.exp_target_data_train == 0)[:, 0]
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            exp_dynamic_pos = torch.FloatTensor(self.exp_dynamic_data_train[pos_idx[:batch_size]])
            exp_dynamic_neg = torch.FloatTensor(self.exp_dynamic_data_train[neg_idx[:batch_size]])
            return exp_dynamic_pos, torch.ones((exp_dynamic_pos.shape[0])), exp_dynamic_neg, torch.zeros((exp_dynamic_neg.shape[0]))
        
        return torch.zeros(0, self.exp_dynamic_data.shape[-1]), torch.ones(0), torch.zeros(0, self.exp_dynamic_data.shape[-1]), torch.zeros(0)

    def update_da(self, exp_dynamic_data, pred_params_data, update):
        """
        Updates the dynamic and target training data if the update flag is set.

        Args:
            exp_dynamic_data (Any): The experimental dynamic data to be updated.
            pred_params_data (Any): The predicted parameters data to be updated.
            update (bool): A flag indicating whether to perform the update.

        Sets:
            self.exp_dynamic_data_train: Updated with exp_dynamic_data if update is True.
            self.exp_target_data_train: Updated with pred_params_data if update is True.
            self.train_available (bool): Set to True if update is True.
        """
        if update:
            self.exp_dynamic_data_train = exp_dynamic_data
            self.exp_target_data_train = pred_params_data
            self.train_available = True