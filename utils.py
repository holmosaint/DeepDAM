import os
import numpy as np
import h5py
import torch
import torch.utils.data as Data
import torch.nn as nn
from collections import OrderedDict
from scipy.stats import zscore


class MyDataset(Data.Dataset):
    """
    A custom PyTorch Dataset for loading data from an HDF5 file.

    Attributes:
        data_path (str): Path to the HDF5 file containing the dataset.
        dynamic_key (str): Key to access the dynamic features in the HDF5 file.
        feature_key (str): Key to access the static features in the HDF5 file.
        target_key (str): Key to access the target values in the HDF5 file.
        offset (int): Offset to start reading data from the HDF5 file.
        size (int): Number of samples in the dataset.

    Methods:
        __init__(data_path, dynamic_key, feature_key, target_key, offset, size):
            Initializes the dataset with the given parameters.
        
        __getitem__(index):
            Retrieves the dynamic features and target values for the given index.
        
        __len__():
            Returns the total number of samples in the dataset.
    """

    def __init__(self, data_path, dynamic_key, feature_key, target_key, offset, size):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.dynamic_key = dynamic_key
        self.feature_key = feature_key
        self.target_key = target_key
        self.offset = offset
        self.size = size

    def __getitem__(self, index):
        id = index + self.offset
        with h5py.File(self.data_path, 'r') as f:
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


class SubNetDataset(Data.Dataset):
    """
    A custom dataset class for loading and processing data from HDF5 files.

    Attributes:
        data_path (str): Path to the HDF5 file containing the data.
        dynamic_key (str): Key to access the dynamic data in the HDF5 file.
        stimulus_key (str): Key to access the stimulus data in the HDF5 file.
        target_key (str): Key to access the target data in the HDF5 file.
        offset (int): Offset value for indexing.
        size (int): Size of the dataset.
        method (str): Method for processing the dynamic data. Options are 'zscore', 'aug', or 'raw'.
        min_thre (float): Minimum threshold for filtering target data.

    Methods:
        __init__(data_path, dynamic_key, stimulus_key, target_key, offset, size, method, min_thre):
            Initializes the dataset with the given parameters and loads the data from the HDF5 file.
        
        __getitem__(index):
            Retrieves the dynamic and target data for the given index. Applies the specified processing method to the dynamic data.
        
        __len__():
            Returns the size of the dataset.
    """

    def __init__(self, data_path, dynamic_key, stimulus_key, target_key, offset,
                 size, method, min_thre):
        super(SubNetDataset, self).__init__()
        self.data_path = data_path
        self.dynamic_key = dynamic_key
        self.stimulus_key = stimulus_key
        self.target_key = target_key
        self.offset = offset
        self.size = size
        self.method = method
        try:
            with h5py.File(self.data_path, 'r') as f:
                G = f.get(self.target_key)[...]
                self.size = G.shape[0]
                self.pos_idx = np.argwhere(G > min_thre)[:, 0]
                self.neg_idx = np.argwhere(G == 0)[:, 0]
        except:
            print('here')
            self.pos_idx = np.arange(self.offset, self.offset+self.size)
            self.neg_idx = np.arange(self.offset, self.offset+self.size)

    def __getitem__(self, index):

        dynamic = np.array([np.nan])
        target = 1e-5
        while np.argwhere(np.isnan(dynamic)).shape[0] > 0:# or np.abs(target) < 1e-3:
            with h5py.File(self.data_path, 'r') as f:

                if np.random.uniform(0, 1) < 0.5:
                    idx = np.random.choice(self.pos_idx, 1)
                else:
                    idx = np.random.choice(self.neg_idx, 1)

                if self.dynamic_key is not None:
                    dynamic = f.get(self.dynamic_key)[idx].astype(
                        np.float32)
                else:
                    dynamic = np.zeros((1)).astype(np.float32)

                if self.target_key is not None:
                    target = f.get(self.target_key)[idx].astype(
                        np.float32)
                else:
                    target = np.zeros((1)).astype(np.float32)
                target[target != 0] = 1

            if self.method == 'zscore':
                dynamic = zscore(dynamic, axis=-1)
            elif self.method == 'aug':
                dynamic = dynamic * np.random.uniform(0.25, 2)
            elif self.method == 'raw':
                dynamic = dynamic

            if len(dynamic.shape) <= 1:
                dynamic = dynamic[np.newaxis]

        return dynamic, target

    def __len__(self):
        return self.size


def save_checkpoint(store_dir, file_name, epoch, model_state_dict,
                    opt_state_dict, train_loss_list, val_loss_list):
    """
    Saves the training checkpoint to a specified directory.

    Args:
        store_dir (str): The directory where the checkpoint will be saved.
        file_name (str): The name of the checkpoint file.
        epoch (int): The current epoch number.
        model_state_dict (dict): The state dictionary of the model.
        opt_state_dict (dict): The state dictionary of the optimizer.
        train_loss_list (list): A list of training loss values.
        val_loss_list (list): A list of validation loss values.

    Returns:
        None
    """
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': opt_state_dict,
            'train_loss': train_loss_list,
            'val_loss': val_loss_list,
        }, os.path.join(store_dir, file_name))


def load_checkpoint(model_path, parallel=True):
    """
    Load a model checkpoint from a given file path.

    Args:
        model_path (str): The path to the model checkpoint file.
        parallel (bool, optional): If True, assumes the model was trained with DataParallel 
                                   and adjusts the state dictionary accordingly. Defaults to True.

    Returns:
        dict: The loaded checkpoint containing the model state dictionary and other metadata.
    """
    checkpoint = torch.load(model_path)
    if not parallel:
        new_checkpoint = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_checkpoint[name] = v
        checkpoint['model_state_dict'] = new_checkpoint
    return checkpoint


def init_lstm_hidden(hidden_dim, batch, layer, _cuda):
    """
    Initializes the hidden state and cell state for an LSTM.

    Args:
        hidden_dim (int): The number of features in the hidden state.
        batch (int): The batch size.
        layer (int): The number of recurrent layers.
        _cuda (bool): If True, moves the parameters to GPU.

    Returns:
        tuple: A tuple containing:
            - document_rnn_init_h (torch.nn.Parameter): Initialized hidden state.
            - document_rnn_init_c (torch.nn.Parameter): Initialized cell state.
    """
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


def construct_dataloader(data_size, data_path, dynamic_key, feature_key, target_key,
                         batch_size, subdata, method, min_thre):
    """
    Constructs and returns training and validation data loaders.

    Parameters:
    - data_size (int): Total size of the dataset.
    - data_path (str): Path to the dataset.
    - dynamic_key (str): Key for dynamic data.
    - feature_key (str): Key for feature data.
    - target_key (str): Key for target data.
    - batch_size (int): Number of samples per batch to load.
    - subdata (bool): Flag to determine if SubNetDataset should be used.
    - method (str): Method to be used in SubNetDataset.
    - min_thre (float): Minimum threshold to be used in SubNetDataset.

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    """
    # Split dataset to train and val
    train_data_size = int(data_size * 0.999)
    val_data_size = data_size - train_data_size

    if not subdata:
        train_dataset = MyDataset(data_path, dynamic_key, feature_key, target_key, 0,
                                  train_data_size)
        val_dataset = MyDataset(data_path, dynamic_key, feature_key, target_key,
                                train_data_size, val_data_size)
    else:
        train_dataset = SubNetDataset(data_path, dynamic_key, feature_key, target_key, 0,
                                      train_data_size, method, min_thre)
        val_dataset = SubNetDataset(data_path, dynamic_key, feature_key, target_key,
                                    train_data_size, val_data_size, method, min_thre)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=1, shuffle=True)
                                            #    sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=1, shuffle=True
                                             )
                                            #  sampler=val_sampler)

    return train_loader, val_loader


def init_weights(m):
    """
    Initialize the weights of a given layer using Xavier uniform initialization.

    Parameters:
    m (torch.nn.Module): The layer to initialize. This function specifically 
                         checks if the layer is an instance of nn.Linear or nn.Conv1d.

    Returns:
    None
    """
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


def write_scalar(writer, key, val, idx):
    """
    Logs a scalar value to the provided writer.

    Parameters:
    writer (SummaryWriter): The writer object used to log the scalar value. If None, the function does nothing.
    key (str): The name of the scalar to log.
    val (float): The scalar value to log.
    idx (int): The index or step at which the scalar is logged.

    Returns:
    None
    """
    if writer is not None:
        writer.add_scalar(key, val, idx)
        writer.flush()


def mse(x, y, w=None):
    """
    Computes the Mean Squared Error (MSE) between two tensors.

    Parameters:
    x (torch.Tensor): The first input tensor.
    y (torch.Tensor): The second input tensor, must be the same shape as `x`.
    w (torch.Tensor, optional): An optional weight tensor. If provided, it should be a 1D tensor 
                                with the same length as the first dimension of `x` and `y`.

    Returns:
    torch.Tensor: The mean squared error. If `w` is provided, returns the weighted mean squared error.
    """
    err = (x - y) ** 2
    if w is not None:
        return torch.mean(err * w[:, None])
    return torch.mean(err)