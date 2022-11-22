import numpy as np
import torch
from torch.distributed import broadcast_multigpu

class ST:
    def __init__(self, active, st_factor, exp_dynamic_data, exp_feature_data, exp_feature_ratio, param_ranges, thre, loss_func, seq_num, bank_size, exp_params=None, exp_score=None, exp_feature=None, syn_dynamic_data=None):
        self.active = active
        self.st_factor = st_factor

        self.bank_size = bank_size
        self.exp_dynamic_data = exp_dynamic_data
        if len(self.exp_dynamic_data.shape) == 2:
            self.exp_dynamic_data = self.exp_dynamic_data[:, np.newaxis]
        self.exp_feature_data = exp_feature_data
        self.exp_feature_ratio = exp_feature_ratio
        
        if exp_params is None:
            self.exp_params = np.zeros((self.exp_dynamic_data.shape[0], self.bank_size, param_ranges.shape[-1]), dtype=np.float32)
            self.exp_score = np.ones((self.exp_dynamic_data.shape[0], self.bank_size), dtype=np.float32) * np.inf
            self.exp_feature = np.zeros((self.exp_dynamic_data.shape[0], self.bank_size, seq_num, exp_feature_data.shape[-1]), dtype=np.float32)
            self.syn_dynamic_data = np.zeros((self.exp_dynamic_data.shape[0], self.bank_size, seq_num, self.exp_dynamic_data.shape[-1]), dtype=np.float32)
        else:
            self.exp_params = exp_params
            self.exp_score = exp_score
            self.exp_feature = exp_feature
            self.syn_dynamic_data = syn_dynamic_data

        self.param_ranges = param_ranges
        self.thre = thre
        self.loss_func = loss_func
        self.seq_num = seq_num

    def cal_good(self, score, thre=None):
        if thre is None:
            thre = self.thre
        return np.argwhere(score < thre)[:, 0]

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.exp_dynamic_data.shape[0]
        else:
            batch_size = min(batch_size, self.exp_dynamic_data.shape[0])
        if self.active:
            sample_idx = self.cal_good(self.exp_score[:, 0])
            np.random.shuffle(sample_idx)
            sample_idx = sample_idx[:batch_size]
            sample_dynamic = torch.FloatTensor(self.exp_dynamic_data[sample_idx])
            sample_param = torch.FloatTensor((self.exp_params[sample_idx, 0] - self.param_ranges[:1]) / (self.param_ranges[1:] - self.param_ranges[:1]))
            return sample_dynamic, sample_param
        return torch.zeros(0, self.seq_num, self.exp_dynamic_data.shape[-1]), torch.zeros(0, self.exp_params.shape[-1])
    
    def sample_pair(self, batch_size):
        # sample for domain adaptation
        batch_size = min(batch_size, self.exp_dynamic_data.shape[0])
        good_idx = self.cal_good(self.exp_score[:, 0])
        np.random.shuffle(good_idx)
        good_idx = good_idx[:batch_size]
        exp_dynamic = torch.FloatTensor(self.exp_dynamic_data[good_idx])
        syn_dynamic = torch.FloatTensor(self.syn_dynamic_data[good_idx, :])
        return exp_dynamic, syn_dynamic
        
    def loss(self, x, y, w):
        if self.active and x.shape[0] > 0:
            return self.loss_func(x, y, w) * self.st_factor
        return torch.zeros(1, device=x.device)

    def score(self, pred_feature_arr):
        s = np.mean(np.abs(pred_feature_arr - self.exp_feature_data[:, None]) / self.exp_feature_ratio, axis=-1)
        s[np.isnan(s)] = 1e8
        return s
    
    def update(self, pred_params_arr, pred_dynamic_arr, pred_feature_arr=None, score=None, cuda=True):
        # Synchronize
        self.exp_score = torch.FloatTensor(self.exp_score)
        self.exp_params = torch.FloatTensor(self.exp_params)
        if cuda:
            self.exp_score = self.exp_score.cuda()
            exp_params = self.exp_params.cuda()
        self.exp_score = [self.exp_score]
        self.exp_params = [exp_params]
        broadcast_multigpu(self.exp_score, 0)
        broadcast_multigpu(self.exp_params, 0)
        self.exp_score = self.exp_score[0].cpu().numpy()
        self.exp_params = self.exp_params[0].cpu().numpy()

        pre_good_idx = self.cal_good(self.exp_score)

        # Update Top BANKSIZE, merge sort
        if score is None:
            score = self.score(pred_feature_arr)    # exp_size*syn_size 
        score_top_idx = np.argsort(score, axis=1)[:, :self.bank_size]
        score_top_val = np.sort(score, axis=1)[:, :self.bank_size]

        exp_score = np.copy(self.exp_score)
        exp_params = np.copy(self.exp_params)
        syn_dynamic_data = np.copy(self.syn_dynamic_data)
        exp_feature = np.copy(self.exp_feature)
        update_size = 0
        for i in range(self.exp_dynamic_data.shape[0]):
            idx, p_i, s_i = 0, 0, 0 # index of pred and current syn data
            while idx < self.bank_size:
                if p_i < pred_dynamic_arr.shape[0] and (s_i >= self.bank_size or exp_score[i, s_i] > score_top_val[i, p_i]):
                    self.exp_score[i, idx] = score_top_val[i, p_i]
                    self.exp_params[i, idx] = pred_params_arr[score_top_idx[i, p_i]]
                    self.syn_dynamic_data[i, idx] = pred_dynamic_arr[score_top_idx[i, p_i]]
                    self.exp_feature[i, idx] = pred_feature_arr[score_top_idx[i, p_i]]
                    p_i += 1
                    update_size += 1
                else:
                    self.exp_score[i, idx] = exp_score[i, s_i] 
                    self.exp_params[i, idx] = exp_params[i, s_i] 
                    self.syn_dynamic_data[i, idx] = syn_dynamic_data[i, s_i] 
                    self.exp_feature[i, idx] = exp_feature[i, s_i] 
                    s_i += 1
                idx += 1

        good_idx = self.cal_good(score_top_val)

        return pre_good_idx.shape[0], self.cal_good(score).shape[0], np.mean(score[good_idx]), update_size
