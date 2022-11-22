import torch
from torch.nn import functional as F
from math import ceil
from scipy.optimize import linear_sum_assignment

def to_onehot(label, num_classes):
    identity = torch.eye(num_classes).to(device=label.device)
    onehot = torch.index_select(identity, 0, label)
    return onehot

class CAN():

    def __init__(self, dataloader_src, dataloader_tgt, n_layer, layer_offset, n_kernel, kernel_mul, thre, st_thre, eps, intra_only, num_classes=2, cuda=True):
        self.dataloader_src = dataloader_src
        self.dataloader_tgt = dataloader_tgt

        self.n_layer = n_layer  # Number of layers as feature
        self.layer_offset = layer_offset
        self.num_classes = num_classes
        self.n_kernel = n_kernel
        self.kernel_mul = kernel_mul

        self.thre = thre
        self.st_thre = st_thre
        self.eps = eps
        self.intra_only = intra_only

        self._cuda = cuda
        self.max_len = 1000

        self.dist = DIST()
        self.cdd = CDD(n_kernel, kernel_mul, num_classes, intra_only=intra_only)

    def get_target_data(self, feature_extractor, regressioner):
        feature_list = list()
        target_list = list()
        dynamic_list = list()
        with torch.no_grad():
            for dynamic_sample, target_sample in self.dataloader_tgt:
                if self._cuda:
                    dynamic_sample = dynamic_sample.cuda(non_blocking=True)
                    target_sample = target_sample.cuda(non_blocking=True)
                
                domain_sample = torch.ones(dynamic_sample.shape[0]).to(dynamic_sample.device)
                feature = feature_extractor(dynamic_sample, domain_sample)
                prediction, fc_feature = regressioner(feature)

                feature = [feature] + fc_feature    # Shape: B * L
                feature = feature[self.layer_offset]
                feature /= torch.norm(feature, p=2, dim=-1).unsqueeze(1)
                feature_list.append(feature)
                target_list.append(target_sample)
                dynamic_list.append(dynamic_sample)

        return torch.cat(dynamic_list, dim=0), torch.cat(target_list, dim=0), torch.cat(feature_list, dim=0)

    def init_cluster(self, feature_extractor, regressioner):
        # Calculate the class centers for source data

        feature_extractor.eval()
        regressioner.eval()

        centers = 0
        refs = torch.FloatTensor(range(self.num_classes)).unsqueeze(1).to(device='cuda' if self._cuda else 'cpu')
        max_iter = 5
        n_iter = 0
        n_mask = 0
        with torch.no_grad():
            for dynamic_sample, target_sample in self.dataloader_src:
                n_iter += 1
                if n_iter > max_iter:
                    break

                if self._cuda:
                    dynamic_sample = dynamic_sample.cuda(non_blocking=True)
                    target_sample = target_sample.cuda(non_blocking=True)

                domain_sample = torch.zeros(dynamic_sample.shape[0]).to(dynamic_sample.device)
                feature = feature_extractor(dynamic_sample, domain_sample)
                prediction, fc_feature = regressioner(feature)

                feature = [feature] + fc_feature    # Shape: B * L
                feature = feature[self.layer_offset]
                feature /= torch.norm(feature, p=2, dim=-1).unsqueeze(1)
                feature = feature.unsqueeze(1)      # Shape: B * 1 * L
                mask = (target_sample.unsqueeze(1) == refs).type(torch.FloatTensor).to(device=feature.device)  # Shape: B * C * 1
                centers += torch.sum(torch.bmm(mask, feature), dim=0)
                n_mask += mask

        feature_extractor.train()
        regressioner.train()
        n_mask = torch.sum(n_mask, dim=0)

        return centers, n_mask  # shape: C * L, C * 1
    
    def clustering_stop(self, tgt_centers, src_centers):
        # centers shape: C * L
        if tgt_centers is None:
            return False
        
        dist = self.dist.get_dist(tgt_centers, src_centers, cross=False)    # shape: C
        dist = torch.mean(dist, dim=0)

        return dist.item() < self.eps

    def assign_label(self, feature, centers):
        dists = self.dist.get_dist(feature, centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels
    
    def align_centers(self, centers, init_centers):
        cost = self.dist.get_dist(centers, init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        # print('col ind', col_ind)
        return col_ind

    def feature_clustering(self, feature_extractor, regressioner, feature):
        centers = None

        refs = torch.LongTensor(range(self.num_classes)).unsqueeze(1).to(device=feature.device)
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        pre_centers, n_mask_src = self.init_cluster(feature_extractor, regressioner)
        init_centers = torch.clone(pre_centers)

        while True:
            if self.clustering_stop(centers, pre_centers):
                break

            if centers is not None:
                pre_centers = centers
            
            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_label(cur_feature, pre_centers)
                labels_onehot = to_onehot(labels, self.num_classes).to(device=cur_feature.device)
                count += torch.sum(labels_onehot, dim=0)    # shape: C
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.FloatTensor).to(device=cur_feature.device)  # shape: C * N
                reshaped_feature = cur_feature.unsqueeze(0)

                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)    # shape: C * N * L -> C * L
                start += cur_len
            
            mask = (count.unsqueeze(1) > 0).type(torch.FloatTensor).to(device=feature.device)
            centers = mask * centers + (1 - mask) * init_centers    # update centers if the new class number is larger than 1, otherwise using init centers (src centers)
        
        dist2center, labels = list(), list()
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_label(cur_feature, centers)

            labels_onehot = to_onehot(cur_labels, self.num_classes).to(device=cur_feature.device)
            count += torch.sum(labels_onehot, dim=0)    # shape: C

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len
        
        labels = torch.cat(labels, dim=0)
        dist2center = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers(centers, init_centers)
        # reorder the centers
        centers = centers[cluster2label]
        # re-label the data according to the index
        for k in range(num_samples):
            labels[k] = cluster2label[labels[k]].item()
        
        center_change = torch.mean(self.dist.get_dist(centers, init_centers, cross=False))

        return centers, labels, center_change, dist2center

    def filter_samples(self, dist2center, thre=None):
        if thre is None:
            thre = self.thre
        min_dist = torch.min(dist2center, dim=1)[0]
        mask = min_dist < self.thre

        return mask

    def filter_class(self):
        pass

    def filtering(self, dist2center, thre=None):
        mask = self.filter_samples(dist2center, thre)

        return mask

    def update_data(self, feature_extractor, regressioner):
        # Clustering and Update Training Data
        with torch.no_grad():
            init_centers, _ = self.init_cluster(feature_extractor, regressioner)
            dynamic_data, target_data, feature_data = self.get_target_data(feature_extractor, regressioner)
            centers, pred_labels_full, center_change, dist2center = self.feature_clustering(feature_extractor, regressioner, feature_data)

            # filter data
            mask = self.filter_samples(dist2center, thre=self.thre)
            if torch.where(mask)[0].shape[0] == 0:
                st_mask = torch.zeros(0, dtype=torch.long)
            else:
                st_mask = self.filter_samples(dist2center[mask], thre=self.st_thre)
            dynamic_data_list = list()
            for i in range(mask.shape[0]):
                if mask[i].item() == 1:
                    dynamic_data_list.append(dynamic_data[i:i+1])
            if len(dynamic_data_list) > 0:
                dynamic_data = torch.cat(dynamic_data_list, dim=0)
            else:
                dynamic_data = torch.zeros((0, 1, dynamic_data.shape[-1])).to(device=dynamic_data.device)
            pred_labels = torch.masked_select(pred_labels_full, mask)
            num_target = pred_labels.shape[0]

        # Sample Source Data
        dynamic_data_src, target_data_src = list(), list()
        num_src = 0
        for dynamic_sample, target_sample in self.dataloader_src:
            dynamic_data_src.append(dynamic_sample.to(device=dynamic_data.device))
            target_data_src.append(target_sample.to(device=dynamic_data.device))
            num_src += dynamic_sample.shape[0]
            if num_src >= num_target:
                break
        dynamic_data_src = torch.cat(dynamic_data_src, dim=0)
        target_data_src = torch.cat(target_data_src, dim=0)

        return dynamic_data_src, target_data_src, dynamic_data, pred_labels, st_mask.cpu().numpy()

    def calc_loss(self, feature_src, label_src, feature_tgt, label_tgt):   
        # Calc Loss
        num_src, num_tgt = list(), list()
        idx_src, idx_tgt = list(), list()
        for i in range(self.num_classes):
            _idx = (label_src == i).nonzero()[:, 0]
            idx_src.append(_idx)
            num_src.append(_idx.shape[0])
            
            _idx = (label_tgt == i).nonzero()[:, 0]
            idx_tgt.append(_idx)
            num_tgt.append(_idx.shape[0])
       
        idx_src = torch.cat(idx_src)
        idx_tgt = torch.cat(idx_tgt)
        feature_src = [feature_src[i][idx_src] for i in range(len(feature_src))]
        feature_tgt = [feature_tgt[i][idx_tgt] for i in range(len(feature_tgt))]

        return self.cdd.forward(feature_src, feature_tgt, num_src, num_tgt)

class DIST:
    def __init__(self):
        pass

    def get_dist(self, pA, pB, cross=False):
        return self.cos(pA, pB, cross)
    
    def cos(self, pA, pB, cross):
        pA = F.normalize(pA, dim=1)
        pB = F.normalize(pB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pA * pB, dim=1))
        else:
            return 0.5 * (1.0 - torch.matmul(pA, pB.transpose(0, 1)))

class CDD(object):
    def __init__(self, kernel_num, kernel_mul, 
                 num_classes, intra_only=False):
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes==1)
    
    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list += [dist_c]
        return dist_list

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + \
	    	2 * torch.sum(dist['st'])

        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N 
        return gamma

    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        assert(len(nums_S) == len(nums_T))
        num_classes = len(nums_S)

        patch = {}
        gammas = {}
        gammas['st'] = torch.zeros_like(dist['st'], requires_grad=False).to(device=dist['st'].device)
        gammas['ss'] = [] 
        gammas['tt'] = [] 
        for c in range(num_classes):
            gammas['ss'] += [torch.zeros([num_classes], requires_grad=False).to(device=dist['st'].device)]
            gammas['tt'] += [torch.zeros([num_classes], requires_grad=False).to(device=dist['st'].device)]

        source_start = source_end = 0
        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_S[ns]
            patch['ss'] = dist['ss'][ns]

            target_start = target_end = 0
            for nt in range(num_classes):
                target_start = target_end 
                target_end = target_start + nums_T[nt] 
                patch['tt'] = dist['tt'][nt]

                patch['st'] = dist['st'].narrow(0, source_start, 
                       nums_S[ns]).narrow(1, target_start, nums_T[nt]) 

                gamma = self.gamma_estimation(patch)

                gammas['ss'][ns][nt] = gamma
                gammas['tt'][nt][ns] = gamma
                gammas['st'][source_start:source_end, \
                     target_start:target_end] = gamma

        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = torch.stack(gamma_list, dim=0).to(device=dist.device)

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps 
        gamma_tensor = gamma_tensor.detach()

        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        num_layers = len(dist_layers)
        kernel_dist = None
        for i in range(num_layers):

            dist = dist_layers[i][key] if category is None else \
                      dist_layers[i][key][category]

            gamma = gamma_layers[i][key] if category is None else \
                      gamma_layers[i][key][category]

            cur_kernel_num = self.kernel_num# [i]
            cur_kernel_mul = self.kernel_mul# [i]

            if kernel_dist is None:
                kernel_dist = self.compute_kernel_dist(dist, 
			gamma, cur_kernel_num, cur_kernel_mul) 

                continue

            kernel_dist += self.compute_kernel_dist(dist, gamma, 
                  cur_kernel_num, cur_kernel_mul) 

        return kernel_dist

    def patch_mean(self, nums_row, nums_col, dist):
        assert(len(nums_row) == len(nums_col))
        num_classes = len(nums_row)

        mean_tensor = torch.zeros([num_classes, num_classes]).to(device=dist.device)
        row_start = row_end = 0
        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]
                val = torch.mean(dist.narrow(0, row_start, 
                           nums_row[row]).narrow(1, col_start, nums_col[col]))
                mean_tensor[row, col] = val
        return mean_tensor
        
    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand))**2).sum(2)
        return dist

    def forward(self, source, target, nums_S, nums_T):
        assert(len(nums_S) == len(nums_T)), \
             "The number of classes for source (%d) and target (%d) should be the same." \
             % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)

        # compute the dist 
        dist_layers = []
        gamma_layers = []

        for i in range(len(source)):

            cur_source = source[i]
            cur_target = target[i]

            dist = {}
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            dist['st'] = self.compute_paired_dist(cur_source, cur_target)

            dist['ss'] = self.split_classwise(dist['ss'], nums_S)
            dist['tt'] = self.split_classwise(dist['tt'], nums_T)
            dist_layers += [dist]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]

        # compute the kernel dist
        for i in range(len(source)):
            for c in range(num_classes):
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)

        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'st')
        kernel_dist_st = self.patch_mean(nums_S, nums_T, kernel_dist_st)

        kernel_dist_ss = []
        kernel_dist_tt = []
        for c in range(num_classes):
            kernel_dist_ss += [torch.mean(self.kernel_layer_aggregation(dist_layers, 
                             gamma_layers, 'ss', c).view(num_classes, -1), dim=1)]
            kernel_dist_tt += [torch.mean(self.kernel_layer_aggregation(dist_layers, 
                             gamma_layers, 'tt', c).view(num_classes, -1), dim=1)]

        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)
        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / self.num_classes

        inter = None
        if not self.intra_only:
            inter_mask = (torch.ones([num_classes, num_classes]) \
                    - torch.eye(num_classes)).type(torch.BoolTensor)
            inter_mask = inter_mask.to(device=mmds.device)
            inter_mmds = torch.masked_select(mmds, inter_mask)
            inter = torch.sum(inter_mmds) / (self.num_classes * (self.num_classes - 1))

        cdd = intra if inter is None else intra - inter
        return {'cdd': cdd, 'intra': intra, 'inter': inter}
