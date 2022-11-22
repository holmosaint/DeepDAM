import os
import numpy as np
import h5py
import torch
import torch.utils.data as Data
import torch.nn as nn
from collections import OrderedDict

class MyDataset(Data.Dataset):

    def __init__(self, data_path, dynamic_key, feature_key, target_key, offset, size):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.dynamic_key = dynamic_key
        self.feature_key = feature_key
        self.target_key = target_key
        self.size = size
        self.offset = offset

    def __getitem__(self, index):
        succ = False
        while not succ:
            id = index + self.offset
            with h5py.File(self.data_path, 'r') as f:
                dynamic = f.get(self.dynamic_key)[id].astype(np.float32)
                feature = f.get(self.feature_key)[id].astype(np.float32)
                target = f.get(self.target_key)[id].astype(np.float32)
            if np.argwhere(np.isnan(dynamic)).shape[0] > 0 or np.argwhere(
                    dynamic.min() < -200).shape[0] > 0 or np.argwhere(
                        dynamic.max() > 150).shape[0] > 0:
                index = (index + 1) % self.file_size
            else:
                succ = True
        return dynamic, feature, target

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


def load_data(store_dir, data_file='data.h5', load_dynamic=True, stimuli=False):
    hf = h5py.File(os.path.join(store_dir, data_file), 'r')
    if not stimuli:
        static_arr = hf.get('static_data').value
    else:
        stimuli_arr = None

    if load_dynamic:
        dynamic_arr = hf.get('dynamic_data').value
    else:
        dynamic_arr = None

    target_arr = hf.get('target_data').value

    if not stimuli:
        return static_arr, dynamic_arr, target_arr

    return stimuli_arr, dynamic_arr, target_arr


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


def SNR(sig, noise):
    Signal = np.var(sig, axis=-1)
    Noise = np.var(noise, axis=-1)
    return 10 * np.log10(Signal / Noise)


def noise_SNR(sig, snr):
    Signal = np.var(sig, axis=-1)
    snr_lin = 10.0**(snr / 10.0)
    pnoise = Signal / snr_lin
    noise = np.sqrt(pnoise[..., np.newaxis]) * np.random.normal(
        0, 1.0, sig.shape)
    return noise


def construct_dataloader(data_size, data_path, dynamic_key, feature_key, target_key,
                         batch_size, subdata):
    # Split dataset to train and val
    train_data_size = int(data_size * 1)
    val_data_size = int(train_data_size * 0.01)

    # Parallel training
    if not subdata:
        train_dataset = MyDataset(data_path, dynamic_key, feature_key, target_key, 0,
                                  train_data_size)
        val_dataset = MyDataset(data_path, dynamic_key, feature_key, target_key,
                                train_data_size, val_data_size)
    else:
        train_dataset = SubNetDataset(data_path, dynamic_key, feature_key, target_key, 0,
                                      train_data_size)
        val_dataset = SubNetDataset(data_path, dynamic_key, feature_key, target_key,
                                    train_data_size, val_data_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
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
        # torch.nn.init.kaiming_normal_(m.weight)
    # m.bias.data.fill_(0.01)


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
