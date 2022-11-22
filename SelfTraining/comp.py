import numpy as np
import torch

class ST:
    def __init__(self, active, st_factor, exp_dynamic_data, exp_target_data, thre):
        self.active = active
        self.st_factor = st_factor

        self.exp_dynamic_data = exp_dynamic_data
        self.exp_target_data = exp_target_data
        self.thre = thre

        self.exp_dynamic_data_train, self.exp_target_data_train = None, None
        self.train_available = False

    def sample(self, batch_size):
        assert self.active, 'Should be active to sample from SelfTraining'
        
        batch_size = min(batch_size, self.exp_dynamic_data.shape[0])
        dynamic_sample_pos, target_sample_pos, dynamic_sample_neg, target_sample_neg = self.sample_balance(int(batch_size / 2))
        sample_dynamic = torch.cat([dynamic_sample_pos, dynamic_sample_neg], dim=0)
        sample_param = torch.cat([target_sample_pos, target_sample_neg], dim=0)
        idx = np.arange(sample_dynamic.shape[0])
        np.random.shuffle(idx)
        sample_dynamic = sample_dynamic[idx]
        sample_param = sample_param[idx].reshape(-1, 1)

        return sample_dynamic, sample_param

    def sample_balance(self, batch_size):
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

    def update(self, exp_dynamic_data, pred_params_data, update):
        if update:
            self.exp_dynamic_data_train = exp_dynamic_data
            self.exp_target_data_train = pred_params_data
            self.train_available = True
