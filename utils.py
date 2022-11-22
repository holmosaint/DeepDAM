import os
import numpy as np
import h5py as h5
import torch
import torch.utils.data as Data
import torch.nn as nn
from collections import OrderedDict

class ExpDataset(Data.Dataset):

    def __init__(self, data_path, dynamic_key, target_key, offset, size):
        super(ExpDataset, self).__init__()
        self.data_path = data_path
        self.dynamic_key = dynamic_key
        self.target_key = target_key
        self.offset = offset
        self.size = size

    def __getitem__(self, index):
        id = index + self.offset
        with h5.File(self.data_path, 'r') as f:
            if self.dynamic_key is not None:
                dynamic = f.get(self.dynamic_key)[id].astype(np.float32)
            else:
                dynamic = np.zeros((1)).astype(np.float32)

            if self.target_key is not None:
                target = f.get(self.target_key)[id].astype(np.float32)
            else:
                target = np.zeros((1)).astype(np.float32)

            if len(dynamic.shape) <= 1:
                dynamic = dynamic[np.newaxis]

        return dynamic, target

    def __len__(self):
        return self.size


class SynDataset(Data.Dataset):

    def __init__(self, data_path, dynamic_key, target_key, offset,
                 size, min_thre):
        super(SynDataset, self).__init__()
        self.data_path = data_path
        self.dynamic_key = dynamic_key
        self.target_key = target_key
        self.offset = offset
        self.size = size
        self.min_thre = min_thre
        
        with h5.File(self.data_path, 'r') as f:
            G = f.get(self.target_key)[...]
            self.size = G.shape[0]
            self.pos_idx = np.argwhere(G > min_thre)[:, 0]
            self.neg_idx = np.argwhere(G == 0)[:, 0]

    def __getitem__(self, index):

        dynamic = np.array([np.nan])
        target = 1e-5
        while np.argwhere(np.isnan(dynamic)).shape[0] > 0:
            with h5.File(self.data_path, 'r') as f:

                if np.random.uniform(0, 1) < 0.5:
                    idx = np.random.choice(self.pos_idx, 1)
                else:
                    idx = np.random.choice(self.neg_idx, 1)

                if self.dynamic_key is not None:
                    dynamic = f.get(self.dynamic_key)[idx].astype(np.float32)
                else:
                    dynamic = np.zeros((1)).astype(np.float32)

                if self.target_key is not None:
                    target = f.get(self.target_key)[idx].astype(np.float32)
                else:
                    target = np.zeros((1)).astype(np.float32)
                target[target != 0] = 1

            dynamic = dynamic * np.random.uniform(0.25, 2)
            if len(dynamic.shape) <= 1:
                dynamic = dynamic[np.newaxis]

        return dynamic, target

    def __len__(self):
        return self.size


def save_checkpoint(store_dir, file_name, epoch, model_state_dict,
                    opt_state_dict, train_loss_list, val_loss_list, exp_data_list):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': opt_state_dict,
            'train_loss': train_loss_list,
            'val_loss': val_loss_list,
            'exp_data': exp_data_list,
        }, os.path.join(store_dir, file_name))


def load_checkpoint(model_path, parallel=True):
    checkpoint = torch.load(model_path)
    if not parallel:
        new_checkpoint = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            # print('Name:', k)
            name = k[7:]  # remove 'module.' of dataparallel
            new_checkpoint[name] = v
        checkpoint['model_state_dict'] = new_checkpoint
    return checkpoint


def init_lstm_hidden(hidden_dim, batch, layer, _cuda):
    document_rnn_init_h = nn.Parameter(nn.init.xavier_uniform_(
        torch.Tensor(layer, batch, hidden_dim).type(torch.FloatTensor)),
                                       requires_grad=True)
    document_rnn_init_c = nn.Parameter(nn.init.xavier_uniform_(
        torch.Tensor(layer, batch, hidden_dim).type(torch.FloatTensor)),
                                       requires_grad=True)
    if _cuda:
        document_rnn_init_h = document_rnn_init_h.cuda()
        document_rnn_init_c = document_rnn_init_c.cuda()
    return (document_rnn_init_h, document_rnn_init_c)
    return noise


def construct_dataloader(data_size, data_path, dynamic_key, target_key,
                         batch_size, min_thre):
    # Split dataset to train and val
    train_data_size = int(data_size * 0.999)
    val_data_size = data_size - train_data_size

    # Parallel training
    train_dataset = SynDataset(data_path, dynamic_key, target_key, 0,
                                    train_data_size, min_thre)
    val_dataset = SynDataset(data_path, dynamic_key, target_key,
                                train_data_size, val_data_size, min_thre)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=1,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=1,
                                             sampler=val_sampler)

    return train_loader, val_loader


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


def write_scalar(writer, key, val, idx):
    if writer is not None:
        writer.add_scalar(key, val, idx)
        writer.flush()


def echo(batch_size):
    rank = torch.distributed.get_rank()
    print("Hello from rank {} with batch size {}".format(rank, batch_size))
    if rank == 0:
        print("Total {} GPUs running!".format(
            torch.distributed.get_world_size()))
