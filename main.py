import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from sklearn.metrics import matthews_corrcoef, roc_curve, auc
from scipy.stats import zscore
from math import ceil

matplotlib.use('Agg')
import argparse
import time
import h5py as h5

from model import FeatureExtractor, Classifier, Classifier

from DomainAdaptation.comp import DA
from SelfTraining.comp import ST

from utils import save_checkpoint, construct_dataloader, init_weights, write_scalar, MyDataset


def construct_exp(exp_test_data_path, batch_size, method):
    """
    Constructs the experimental setup for testing.
    Parameters:
    exp_test_data_path (str or None): Path to the HDF5 file containing test data. If None, empty arrays are used.
    batch_size (int): The batch size for the DataLoader.
    method (str): Method to preprocess the dynamic data. Currently supports 'zscore'.
    Returns:
    tuple: A tuple containing:
        - exp_dynamic_data_test (numpy.ndarray): The dynamic test data.
        - exp_target_data_test (numpy.ndarray): The target test data.
        - exp_dataloader (torch.utils.data.DataLoader or None): DataLoader for the test data, or None if no test data path is provided.
    """

    if exp_test_data_path is not None:
        with h5.File(exp_test_data_path, 'r') as f:
            exp_dynamic_data_test = f.get('dynamic_data')[...].astype(np.float32)
            if method == 'zscore':
                exp_dynamic_data_test = zscore(exp_dynamic_data_test.reshape(-1, exp_dynamic_data_test.shape[-1]), axis=-1)
            exp_target_data_test = f.get('target_data')[...].reshape(-1)
            exp_target_data_test[exp_target_data_test != 0] = 1
            exp_target_data_test = exp_target_data_test.astype(np.int32)
        exp_dataset = MyDataset(exp_test_data_path, 'dynamic_data', None, 'target_data', 0, exp_dynamic_data_test.shape[0])
        exp_dataloader = torch.utils.data.DataLoader(exp_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    else:
        exp_dynamic_data_test = np.zeros((0, 100))
        exp_target_data_test = np.zeros((0, 100))
        exp_dataloader = None
       
    return exp_dynamic_data_test, exp_target_data_test, exp_dataloader


class RPLLoss(torch.nn.Module):
    """
    RPLLoss is a custom loss function based on the Robust Pseudo-Labeling (RPL) method.

    Args:
        q (float, optional): A hyperparameter that controls the shape of the loss function. Default is 0.5.

    Methods:
        forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Computes the RPL loss between the input and the target tensors.

            Args:
                input (torch.Tensor): The input tensor, typically the output from a neural network.
                target (torch.Tensor): The target tensor, containing the ground truth labels.

            Returns:
                torch.Tensor: The computed RPL loss.
    """
    def __init__(self, q=0.5) -> None:
        super().__init__()
        self.q = q

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.sigmoid(input)
        loss = (1 - target) * (1 - (1 - input) ** self.q) + target * (1 - input ** self.q)
        loss /= self.q
        return torch.mean(loss, dim=0)


def construct_net(
    time_len,
    sequence_num,
    output_dim,
    cnn_type,
    time_type,
    fc_layers,
    cnn_config,
    lr,
    store_dir,
    model_path,
    cuda=True,
    dropout=False,
):
    """
    Constructs the neural network components, initializes weights, and sets up the optimizer and loss functions.

    Args:
        time_len (int): Length of the time dimension.
        sequence_num (int): Number of sequences.
        output_dim (int): Dimension of the output.
        cnn_type (str): Type of CNN to use.
        time_type (str): Type of time representation.
        fc_layers (list): List of fully connected layer configurations.
        cnn_config (dict): Configuration dictionary for the CNN.
        lr (float): Learning rate for the optimizer.
        store_dir (str): Directory to store logs and checkpoints.
        model_path (str or None): Path to the model checkpoint for loading, or None to start fresh.
        cuda (bool, optional): Whether to use CUDA. Defaults to True.
        dropout (bool, optional): Whether to use dropout. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - feature_extractor (nn.Module): The feature extractor network.
            - classifier (nn.Module): The regression network.
            - classifier (nn.Module): The classifier network (or identity if not using DANN).
            - opt_Adam (torch.optim.Optimizer): The Adam optimizer.
            - loss_func (nn.Module): The binary cross-entropy loss function.
            - self_training_loss_func (nn.Module): The self-training loss function.
            - writer (SummaryWriter): The summary writer for TensorBoard logging.
            - train_loss_list (list): List of training losses.
            - train_syn_loss_list (list): List of training synthetic losses.
            - train_da_loss_list (list): List of training domain adaptation losses.
            - train_st_loss_list (list): List of training self-training losses.
            - val_loss_list (list): List of validation losses.
            - val_syn_loss_list (list): List of validation synthetic losses.
            - val_da_loss_list (list): List of validation domain adaptation losses.
            - val_st_loss_list (list): List of validation self-training losses.
            - last_val_loss (float): Last validation loss.
            - last_val_syn_loss (float): Last validation synthetic loss.
            - last_val_da_loss (float): Last validation domain adaptation loss.
            - last_val_st_loss (float): Last validation self-training loss.
            - _epoch (int): Current epoch number.
            - train_idx (int): Training index.
            - val_idx (int): Validation index.
    """
    feature_extractor = FeatureExtractor(
        time_len,
        sequence_num,
        output_dim,
        cnn_type,
        time_type,
        fc_layers,
        cnn_config,
        cuda=cuda,
        dropout=dropout,
    )
    classifier = Classifier(feature_extractor.fc_dim,
                                output_dim,
                                fc_layers,
                                cuda=cuda,
                                dropout=dropout)

    if cuda:
        feature_extractor.cuda()
        classifier.cuda()
        classifier.cuda()

    feature_extractor.apply(init_weights)
    classifier.apply(init_weights)
    opt_Adam = torch.optim.Adam(filter(
        lambda p: p.requires_grad,
        list(feature_extractor.parameters()) + list(classifier.parameters()) +
        list(classifier.parameters())),
                                lr=lr,
                                betas=(0.9, 0.99))
    loss_func = nn.BCEWithLogitsLoss()
    self_training_loss_func = RPLLoss(q=1.0)

    writer = SummaryWriter(os.path.join(store_dir, 'log'))

    if model_path is None:
        train_loss_list = list()
        train_syn_loss_list = list()
        train_da_loss_list = list()
        train_st_loss_list = list()
        val_loss_list = list()
        val_syn_loss_list = list()
        val_da_loss_list = list()
        val_st_loss_list = list()
        last_val_loss = 1e8
        last_val_syn_loss = 1e8
        last_val_da_loss = 1e8
        last_val_st_loss = 1e8
        _epoch = 0
        train_idx = 0
        val_idx = 0
    else:
        checkpoint = torch.load(os.path.join(model_path, 'feature_extractor.h5'))
        train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list = checkpoint['train_loss']
        val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list = checkpoint['val_loss']
        opt_Adam.load_state_dict(checkpoint['optimizer_state_dict'])
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        _epoch = checkpoint['epoch'][-1]
        last_val_loss = min(val_loss_list)
        last_val_syn_loss = min(val_syn_loss_list)
        last_val_da_loss = min(val_da_loss_list)
        last_val_st_loss = min(val_st_loss_list)
        train_idx = len(train_loss_list)
        val_idx = len(val_loss_list)

        checkpoint = torch.load(os.path.join(model_path, 'classifier.h5'))
        classifier.load_state_dict(checkpoint['model_state_dict'])

        for i, train_loss in enumerate(train_loss_list):
            writer.add_scalar('Training Loss', train_loss, i)
        for i, val_loss in enumerate(val_loss_list):
            writer.add_scalar('Validation Loss', val_loss, i)
        for i, train_loss in enumerate(train_syn_loss_list):
            writer.add_scalar('Training Syn Loss', train_loss, i)
        for i, val_loss in enumerate(val_syn_loss_list):
            writer.add_scalar('Validation Syn Loss', val_loss, i)
        for i, train_loss in enumerate(train_da_loss_list):
            writer.add_scalar('Training DA Loss', train_loss, i)
        for i, val_loss in enumerate(val_da_loss_list):
            writer.add_scalar('Validation DA Loss', val_loss, i)
        for i, train_loss in enumerate(train_st_loss_list):
            writer.add_scalar('Training ST Loss', train_loss, i)
        for i, val_loss in enumerate(val_st_loss_list):
            writer.add_scalar('Validation ST Loss', val_loss, i)

    return feature_extractor, classifier, opt_Adam, loss_func, self_training_loss_func, writer, train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list, val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list, last_val_loss, last_val_syn_loss, last_val_da_loss, last_val_st_loss, _epoch, train_idx, val_idx


def eval_exp_data(exp_dynamic_data, feature_extractor, classifier, batch_size, cuda):
    """
    Evaluate experimental data using a feature extractor and regression model.

    Args:
        exp_dynamic_data (np.ndarray): The dynamic experimental data to be evaluated.
        feature_extractor (torch.nn.Module): The feature extractor model.
        classifier (torch.nn.Module): The regression model.
        batch_size (int): The size of each batch for processing the data.
        cuda (bool): Flag indicating whether to use CUDA for GPU acceleration.

    Returns:
        tuple: A tuple containing:
            - pred_param_arr (np.ndarray): The predicted parameters after thresholding.
            - pred_score_arr (np.ndarray): The predicted scores before thresholding.
    """

    pred_param_arr = list()

    for exp_batch_idx, start_idx in enumerate(
            range(0, exp_dynamic_data.shape[0], batch_size)):
        dynamic_sample = torch.FloatTensor(
            exp_dynamic_data[start_idx:min(start_idx +
                                           batch_size, exp_dynamic_data.shape[0]
                                          )])
        batch = dynamic_sample.shape[0]
        if cuda:
            dynamic_sample = dynamic_sample.cuda(non_blocking=True)
        with torch.no_grad():
            prediction, _ = classifier(
                feature_extractor(dynamic_sample))
            prediction = torch.sigmoid(prediction)

        pred_param_arr.append(prediction.detach().cpu().numpy())

    pred_param_arr = np.concatenate(pred_param_arr, axis=0)
    pred_score_arr = np.copy(pred_param_arr[:, 0])
    pred_param_arr[pred_param_arr < 0.5] = 0
    pred_param_arr[pred_param_arr >= 0.5] = 1
    pred_param_arr = pred_param_arr.reshape(-1)

    return pred_param_arr, pred_score_arr


def save_model(store_dir, feature_extractor, classifier, opt, train_loss_list, val_loss_list, epoch):
    """
    Save the model's feature extractor and classifier along with their states and training information.

    Args:
        store_dir (str): Directory where the model checkpoints will be saved.
        feature_extractor (torch.nn.Module): The feature extractor part of the model.
        classifier (torch.nn.Module): The classifier part of the model.
        opt (torch.optim.Optimizer): The optimizer used for training the model.
        exp_params (dict): Experimental parameters used during training.
        exp_score (float): The experimental score achieved by the model.
        train_loss_list (list): List of training losses recorded over epochs.
        val_loss_list (list): List of validation losses recorded over epochs.
        epoch (int): The current epoch number.

    Returns:
        None
    """

    save_checkpoint(store_dir, f'feature_extractor_epoch{epoch}.h5', [0],
                    feature_extractor.state_dict(), opt.state_dict(),
                    train_loss_list, val_loss_list)
    save_checkpoint(store_dir, f'classifier_epoch{epoch}.h5', [0],
                    classifier.state_dict(), opt.state_dict(), [], [])


def train_step(feature_extractor,
               classifier,
               dynamic_sample,
               target_sample,
               domain_adaptation,
               self_training,
               cuda,
               loss_func,
               self_training_loss_func,
               opt=None):
    """
    Perform a single training step for a model with domain adaptation and self-training.
    Args:
        feature_extractor (nn.Module): The feature extractor model.
        classifier (nn.Module): The regression model.
        dynamic_sample (torch.Tensor): Input tensor for synthesized data.
        target_sample (torch.Tensor): Target tensor for synthesized data.
        domain_adaptation (object): Domain adaptation object with an active attribute and sample method.
        self_training (object): Self-training object with an active attribute and sample method.
        cuda (bool): Flag to indicate if CUDA should be used.
        loss_func (callable): Loss function for synthesized data.
        self_training_loss_func (callable): Loss function for self-training data.
        opt (torch.optim.Optimizer, optional): Optimizer for updating model parameters. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - loss (torch.Tensor): Total loss combining synthesized data, domain adaptation, and self-training losses.
            - syn_loss (torch.Tensor): Loss from synthesized data.
            - da_loss (torch.Tensor): Loss from domain adaptation.
            - st_loss (torch.Tensor): Loss from self-training.
            - opt (torch.optim.Optimizer): Optimizer after performing the step.
            - (exp_dynamic_sample, exp_target_sample, st_mask): Tuple containing experimental dynamic sample, target sample, and self-training mask.
    """
    # Synthesized Data Training
    batch_size = dynamic_sample.shape[0]
    if cuda:
        dynamic_sample = dynamic_sample.cuda(non_blocking=True)
        target_sample = target_sample.cuda(non_blocking=True)
    syn_feature = feature_extractor(dynamic_sample)
    syn_prediction, syn_fc_feature = classifier(syn_feature)

    syn_loss = loss_func(syn_prediction, target_sample)
    if opt is not None:
        opt.zero_grad()
        syn_loss.backward()
        opt.step()

    # Self Training
    st_loss = torch.zeros([1]).to(device=syn_loss.device)
    if self_training.active:
        st_dynamic_sample, st_target_sample = self_training.sample(batch_size)
        if cuda:
            st_dynamic_sample = st_dynamic_sample.cuda(non_blocking=True)
            st_target_sample = st_target_sample.cuda(non_blocking=True)
        for i in range(0, st_dynamic_sample.shape[0], batch_size):
            _end = min(i+batch_size, st_dynamic_sample.shape[0])
            feature = feature_extractor(st_dynamic_sample[i:_end])
            prediction, fc_feature = classifier(feature)

            loss = self_training_loss_func(prediction, st_target_sample[i:_end])
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            st_loss += loss * (_end - i)
        if st_dynamic_sample.shape[0] > 0:
            st_loss /= st_dynamic_sample.shape[0]
        st_loss *= self_training.st_factor

    # Domain Adaptation Training
    if domain_adaptation.active:
        syn_dynamic_sample_pos, syn_target_sample_pos, syn_dynamic_sample_neg, syn_target_sample_neg, exp_dynamic_sample_pos, exp_target_sample_pos, exp_dynamic_sample_neg, exp_target_sample_neg, st_mask = domain_adaptation.sample(feature_extractor, classifier)
        if cuda:
            syn_dynamic_sample_pos = syn_dynamic_sample_pos.cuda(non_blocking=True)
            syn_dynamic_sample_neg = syn_dynamic_sample_neg.cuda(non_blocking=True)
            exp_dynamic_sample_pos = exp_dynamic_sample_pos.cuda(non_blocking=True)
            exp_dynamic_sample_neg = exp_dynamic_sample_neg.cuda(non_blocking=True)

        syn_size_pos = syn_dynamic_sample_pos.shape[0]
        syn_size_neg = syn_dynamic_sample_neg.shape[0]
        exp_size_pos = exp_dynamic_sample_pos.shape[0]
        exp_size_neg = exp_dynamic_sample_neg.shape[0]
        n_split = int(ceil(np.min([syn_size_pos, syn_size_neg, exp_size_pos, exp_size_neg]) / batch_size * 4))

        da_loss = torch.zeros([1]).to(device=syn_loss.device)
        syn_feature_list = list()
        syn_target_list = list()

        exp_dynamic_sample = torch.cat([exp_dynamic_sample_pos, exp_dynamic_sample_neg], dim=0).cpu().numpy()
        exp_target_sample = torch.cat([exp_target_sample_pos, exp_target_sample_neg], dim=0).cpu().numpy()

        if n_split > 0:
            syn_step_pos = np.linspace(0, syn_size_pos, num=n_split+1).astype(np.int32)
            syn_step_neg = np.linspace(0, syn_size_neg, num=n_split+1).astype(np.int32)
            exp_step_pos = np.linspace(0, exp_size_pos, num=n_split+1).astype(np.int32)
            exp_step_neg = np.linspace(0, exp_size_neg, num=n_split+1).astype(np.int32)

            for i in range(n_split):
                syn_sample_pos = syn_dynamic_sample_pos[syn_step_pos[i]:syn_step_pos[i+1]]
                syn_sample_neg = syn_dynamic_sample_neg[syn_step_neg[i]:syn_step_neg[i+1]]
                exp_sample_pos = exp_dynamic_sample_pos[exp_step_pos[i]:exp_step_pos[i+1]]
                exp_sample_neg = exp_dynamic_sample_neg[exp_step_neg[i]:exp_step_neg[i+1]]
                _syn_size = syn_sample_pos.shape[0] + syn_sample_neg.shape[0]
                _exp_size = exp_sample_pos.shape[0] + exp_sample_neg.shape[0]

                da_dynamic = torch.cat([syn_sample_pos, syn_sample_neg, exp_sample_pos, exp_sample_neg], dim=0)

                da_feature = feature_extractor(da_dynamic)
                da_S_feature, da_T_feature = da_feature[:_syn_size], da_feature[_syn_size:]

                syn_target = torch.cat([torch.ones(syn_sample_pos.shape[0]), torch.zeros(syn_sample_neg.shape[0])]).to(device=da_dynamic.device)
                exp_target = torch.cat([torch.ones(exp_sample_pos.shape[0]), torch.zeros(exp_sample_neg.shape[0])]).to(device=da_dynamic.device)
                loss = domain_adaptation.loss([da_S_feature], [da_T_feature], syn_target, exp_target)
                if opt is not None:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                da_loss += loss
                syn_feature_list.append(da_S_feature.detach().cpu().numpy())
                syn_target_list.append(syn_target.detach().cpu().numpy())
            
            da_loss /= n_split

    else:
        da_loss = torch.zeros([1]).to(device=syn_loss.device)
        exp_dynamic_sample, exp_target_sample, st_mask = None, None, None

    loss = syn_loss + da_loss + st_loss
    
    return loss, syn_loss, da_loss, st_loss, opt, (exp_dynamic_sample, exp_target_sample, st_mask)


def train(args):
    """
    Train the model with the given arguments.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - t (int): Time length.
            - b (int): Batch size.
            - e (int): Number of epochs.
            - c (bool): Use CUDA if True.
            - dir (str): Directory to store the model.
            - lr (float): Learning rate.
            - syn_data (str): Path to synthetic data.
            - exp_test_data (str): Path to experimental test data.
            - seq (int): Sequence number.
            - model (str): Path to the model.
            - out (int): Output dimension.
            - cnn_type (str): Type of CNN.
            - time_type (str): Type of time representation.
            - fl (list): Fully connected layers configuration.
            - cnn_config (dict): CNN configuration.
            - size (int): Data size.
            - dr (float): Dropout rate.
            - dynamic_key (str): Key for dynamic data.
            - feature_key (str): Key for feature data.
            - target_key (str): Key for target data.
            - subdata (bool): Use subdata if True.
            - start_idx (int): Starting index for training.
            - method (str): Method for data processing.
            - min_thre (float): Minimum threshold.
            - da (bool): Use domain adaptation if True.
            - da_type (str): Type of domain adaptation.
            - da_layer (int): Domain adaptation layer.
            - da_n_layer (int): Number of domain adaptation layers.
            - da_layer_offset (int): Offset for domain adaptation layers.
            - da_n_kernel (int): Number of kernels for domain adaptation.
            - da_kernel_mul (float): Kernel multiplier for domain adaptation.
            - da_factor (float): Factor for domain adaptation.
            - da_thre (float): Threshold for domain adaptation.
            - st_thre (float): Threshold for self-training.
            - da_eps (float): Epsilon for domain adaptation.
            - da_intra_only (bool): Use intra-domain adaptation only if True.
            - st (bool): Use self-training if True.
            - st_factor (float): Factor for self-training.
    Returns:
        None
    """

    time_len = args.t
    batch_size = args.b
    epoch = args.e
    cuda = args.c
    store_dir = args.dir
    lr = args.lr
    data_path = args.syn_data
    exp_test_data_path = args.exp_test_data
    sequence_num = args.seq
    model_path = args.model
    output_dim = args.out
    cnn_type = args.cnn_type
    time_type = args.time_type
    fc_layers = args.fl
    cnn_config = args.cnn_config
    data_size = args.size
    dropout = args.dr
    dynamic_key = args.dynamic_key
    feature_key = args.feature_key
    target_key = args.target_key
    subdata = bool(args.subdata)
    start_idx = args.start_idx
    method = args.method
    min_thre = args.min_thre

    feature_extractor, classifier, opt_Adam, loss_func, self_training_loss_func, writer, train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list, val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list, last_val_loss, last_val_syn_loss, last_val_da_loss, last_val_st_loss, _epoch, train_idx, val_idx = construct_net(
        time_len,
        sequence_num,
        output_dim,
        cnn_type,
        time_type,
        fc_layers,
        cnn_config,
        lr,
        store_dir,
        model_path,
        cuda=cuda,
        dropout=dropout,
    )

    train_loader, val_loader = construct_dataloader(data_size, data_path,
                                                    dynamic_key, feature_key,
                                                    target_key, batch_size,
                                                    subdata, method, min_thre)
    exp_dynamic_data_unknown, exp_target_data_unknown, exp_dataloader = construct_exp(exp_test_data_path, batch_size, method)

    not_change = 0
    last_mcc = 0
    mcc_list = list()
    not_improve = torch.FloatTensor([0])
    if cuda:
        not_improve = not_improve.cuda()

    # Domain Adaptation
    domain_adaptation = DA(args.da, args.da_type, args.da_layer,
                           args.da_n_layer, args.da_layer_offset,
                           args.da_n_kernel, args.da_kernel_mul, args.da_factor, train_loader, exp_dataloader, args.da_thre, args.st_thre, args.da_eps, args.da_intra_only, writer=writer)

    # Self Training
    self_training = ST(args.st, args.st_factor, exp_dynamic_data_unknown, exp_target_data_unknown, args.st_thre)

    is_break = False

    if _epoch == 0:
        save_model(
            store_dir, feature_extractor, classifier, opt_Adam,
            [
                train_loss_list, train_syn_loss_list,
                train_da_loss_list, train_st_loss_list,
            ], [
                val_loss_list, val_syn_loss_list, val_da_loss_list,
                val_st_loss_list,
            ], -1)

    for e in range(_epoch, epoch):
        if is_break:
            break

        for batch_idx, (dynamic_sample, target_sample) in enumerate(train_loader):
            # print("Train Batch: ", batch_idx, end='\r')

            feature_extractor.train()
            classifier.train()

            if batch_idx < start_idx and e == 0:
                batch_size = dynamic_sample.shape[0]
                if cuda:
                    dynamic_sample = dynamic_sample.cuda(non_blocking=True)
                    target_sample = target_sample.cuda(non_blocking=True)
                syn_feature = feature_extractor(dynamic_sample)
                syn_prediction, syn_fc_feature = classifier(syn_feature)

                syn_loss = loss_func(syn_prediction, target_sample)
                opt_Adam.zero_grad()
                syn_loss.backward()
                opt_Adam.step()
   
                continue

            loss, syn_loss, da_loss, st_loss, opt_Adam, exp_da_data = train_step(feature_extractor, classifier, dynamic_sample, target_sample, domain_adaptation, self_training, cuda, loss_func, self_training_loss_func, opt=opt_Adam)

            # Experiment Evaluation
            feature_extractor.eval()
            classifier.eval()

            pred_param_data, pred_score_data = eval_exp_data(exp_dynamic_data_unknown, feature_extractor, classifier, dynamic_sample.shape[0], cuda)
            # Training Log
            train_loss_list.append(loss.item())
            train_syn_loss_list.append(syn_loss.item())
            train_da_loss_list.append(da_loss.item())
            train_st_loss_list.append(st_loss.item())

            # Write Training Log
            write_scalar(writer, 'Training Loss', loss.item(), train_idx)
            write_scalar(writer, 'Training Syn Loss', syn_loss.item(),
                            train_idx)
            write_scalar(writer, 'Training DA Loss', da_loss.item(),
                            train_idx)
            write_scalar(writer, 'Training ST Loss', st_loss.item(),
                            train_idx)
            pred_param_data[pred_param_data > 0.5] = 1
            pred_param_data[pred_param_data <= 0.5] = 0
            write_scalar(writer, 'Epoch Acc', np.argwhere(pred_param_data == exp_target_data_unknown).shape[0] / pred_param_data.shape[0],
                            train_idx)
            write_scalar(writer, 'Epoch TP',np.argwhere(np.logical_and(pred_param_data == exp_target_data_unknown, exp_target_data_unknown == 1)).shape[0] / np.argwhere(exp_target_data_unknown == 1).shape[0], train_idx)
            write_scalar(writer, 'Epoch TN',np.argwhere(np.logical_and(pred_param_data == exp_target_data_unknown, exp_target_data_unknown == 0)).shape[0] / np.argwhere(exp_target_data_unknown == 0).shape[0], train_idx)
            write_scalar(writer, 'Epoch FP',np.argwhere(np.logical_and(pred_param_data != exp_target_data_unknown, exp_target_data_unknown == 0)).shape[0] / np.argwhere(exp_target_data_unknown == 0).shape[0], train_idx)
            write_scalar(writer, 'Epoch FN',np.argwhere(np.logical_and(pred_param_data != exp_target_data_unknown, exp_target_data_unknown == 1)).shape[0] / np.argwhere(exp_target_data_unknown == 1).shape[0], train_idx)

            cur_mcc = matthews_corrcoef(exp_target_data_unknown, pred_param_data)
            write_scalar(writer, 'Epoch MCC', cur_mcc, train_idx)
            print('MCC:', cur_mcc, end='\r')
            mcc_list.append(cur_mcc)
            write_scalar(writer, 'Pos pred size', np.argwhere(pred_param_data == 1).shape[0], train_idx)

            # Use Synthesized Data for Validation
            feature_extractor.eval()
            classifier.eval()
            tmp_loss_list = list()
            tmp_syn_loss_list = list()
            tmp_da_loss_list = list()
            tmp_st_loss_list = list()
            max_iter = 10
            _iter = 0
            with torch.no_grad():
                for val_batch_idx, (dynamic_sample, target_sample) in enumerate(val_loader):
                    _iter += 1
                    if _iter >= max_iter:
                        break
                    domain_adaptation.active = False
                    self_training.active = False
                    loss, syn_loss, da_loss, st_loss, _, _ = train_step(feature_extractor, classifier, dynamic_sample, target_sample, domain_adaptation, self_training, cuda, loss_func, self_training_loss_func, opt=None)
                    domain_adaptation.active = args.da
                    self_training.active = args.st

                    tmp_loss_list.append(loss.item())
                    tmp_syn_loss_list.append(syn_loss.item())
                    tmp_da_loss_list.append(da_loss.item())
                    tmp_st_loss_list.append(st_loss.item())
            val_loss = torch.FloatTensor(
                [np.mean(tmp_loss_list).astype(np.float32)]).cuda()
            val_syn_loss = torch.FloatTensor(
                [np.mean(tmp_syn_loss_list).astype(np.float32)]).cuda()
            val_da_loss = torch.FloatTensor(
                [np.mean(tmp_da_loss_list).astype(np.float32)]).cuda()
            val_st_loss = torch.FloatTensor(
                [np.mean(tmp_st_loss_list).astype(np.float32)]).cuda()
            val_loss_list.append(val_loss)
            val_syn_loss_list.append(val_syn_loss)
            val_da_loss_list.append(val_da_loss)
            val_st_loss_list.append(val_st_loss)
            write_scalar(writer, 'Val Loss', val_loss, val_idx)
            write_scalar(writer, 'Val Syn Loss', val_syn_loss, val_idx)
            write_scalar(writer, 'Val DA Loss', val_da_loss, val_idx)
            write_scalar(writer, 'Val ST Loss', val_st_loss, val_idx)
            val_idx += 1
            
            update = val_loss < last_val_loss
            last_val_loss = min(last_val_loss, val_loss)

            # Update Self Training
            if self_training.active:
                self_training.update_da(exp_da_data[0][exp_da_data[2]], exp_da_data[1][exp_da_data[2]], update=update)

            if not update:
                not_improve += 1
                if not_improve % 10 == 0:
                    for p in opt_Adam.param_groups:
                        p['lr'] /= 2
                if not_improve % 50 == 0:
                    is_break = True

                    save_model(
                        store_dir, feature_extractor, classifier, opt_Adam,
                        [
                            train_loss_list, train_syn_loss_list,
                            train_da_loss_list, train_st_loss_list,
                        ], [
                            val_loss_list, val_syn_loss_list, val_da_loss_list,
                            val_st_loss_list
                        ], train_idx)
                    pickle.dump(mcc_list, open(f'{store_dir}/mcc_list.pkl', 'wb'))
                    break
            else:
                not_improve = 0
                with h5.File(os.path.join(store_dir, 'pred.h5'), 'w') as f:
                    f.create_dataset('pred_param', data=pred_score_data)
                    f.create_dataset('target_param', data=exp_target_data_unknown)
            if cur_mcc == last_mcc:
                not_change += 1
            else:
                not_change = 0
                last_mcc = cur_mcc

            write_scalar(writer, 'Update', int(update), train_idx)
            write_scalar(writer, 'Not Improve', not_improve,
                         train_idx)

            train_idx += 1
            import pickle
            pickle.dump({
                'target_data': exp_da_data[0],
                'pred_data': exp_da_data[1],
                'mask': exp_da_data[2]
            }, open(f'{store_dir}/exp_da_data_{train_idx}.pkl', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Architecture arguments
    parser.add_argument('-seed', help='Random seed', default=0, type=int)
    parser.add_argument('-t', help='Time series length', default=5000, type=int)
    parser.add_argument('-b', help='Batch size', default=32, type=int)
    parser.add_argument('-e', help='Epoch', default=100, type=int)
    parser.add_argument('-c', help='Cuda', default=True, type=bool)
    parser.add_argument('-dir', help='Store directory', default='.', type=str)
    parser.add_argument('-lr', help='Learning rate', default=1e-3, type=float)
    parser.add_argument('-syn_data',
                        help='Path to synthesized data .h5 file',
                        required=True,
                        type=str)
    parser.add_argument('-exp_test_data',
                        help='Path to exp test data',
                        default=None,
                        type=str)
    parser.add_argument('-model',
                        help='Pre-trained Model Path',
                        default=None,
                        type=str)
    parser.add_argument('-size',
                        help='Number of training data',
                        required=True,
                        type=int)
    parser.add_argument('-seq',
                        help='Number of sequences',
                        default=50,
                        type=int)
    parser.add_argument('-out', help='Output Dimension', default=3, type=int)
    parser.add_argument(
        '-cnn_type',
        help=
        'Type of cnn to use: [resnet]',
        type=str,
        required=True)
    parser.add_argument('-cnn_config',
                        help='CNN configuration',
                        type=int,
                        default=-1)
    parser.add_argument(
        '-time_type',
        help='Type of time net to use: [lstm]',
        type=str,
        required=True)
    parser.add_argument('-fl', help='FC layers', default=2, type=int)
    parser.add_argument('-dr', help='Dropout', default=0, type=int)
    parser.add_argument('-dynamic_key',
                        help='dynamic key of the data file',
                        default=None,
                        type=str)
    parser.add_argument('-feature_key',
                        help='feature key of the data file',
                        default=None,
                        type=str)
    parser.add_argument('-target_key',
                        help='target key of the data file',
                        default=None,
                        type=str)
    parser.add_argument('-subdata',
                        help='Whether to use sub net dataset',
                        default=0,
                        type=int)
    parser.add_argument('-start_idx', 
                        help='Which step to start DAST', 
                        default=0,
                        type=int)
    parser.add_argument('-method',
                        help='Preprocessing method',
                        default='zscore',
                        type=str)
    parser.add_argument('-min_thre',
                        help='Threshold for synthesized positive data',
                        required=True,
                        type=float)

    # Domain Adaptation
    parser.add_argument('-da',
                        help='Whether to use domain adaptation',
                        type=int,
                        required=True)
    parser.add_argument(
        '-da_type',
        help=
        'Type of domain adaptation techniques to use: [CAN]',
        type=str,
        default='CAN')
    parser.add_argument('-da_layer',
                        help='Which layer to calc dis loss: [feature, fc]',
                        type=str,
                        default='feature')
    parser.add_argument('-da_n_layer',
                        help='# of layers to calc dis loss',
                        type=int,
                        default=0)
    parser.add_argument('-da_n_kernel',
                        help='# of kernels to use in DAN, > 1',
                        type=int,
                        default=2)
    parser.add_argument('-da_kernel_mul',
                        help='Sigma base for kernels to use in DAN, > 1',
                        type=int,
                        default=2)
    parser.add_argument('-da_layer_offset',
                        help='Layer offset to calculate da loss, > 1',
                        type=int,
                        default=0)
    parser.add_argument('-da_thre',
                        help='Thre for CAN filtering',
                        type=float,
                        default=0)
    parser.add_argument('-da_eps',
                        help='Eps for CAN KNN stopping criterion',
                        type=float,
                        default=0)
    parser.add_argument('-da_intra_only',
                        help='Whether to use intra loss only for CAN',
                        type=int,
                        default=0)
    parser.add_argument('-da_factor',
                        help='Factor of domain adaptation loss',
                        type=float,
                        default=0)

    # Self Training
    parser.add_argument('-st',
                        help='Whether to use self training',
                        type=int,
                        required=True)
    parser.add_argument('-st_factor',
                        help='Factor of exp training loss',
                        type=float,
                        default=0.0)
    parser.add_argument('-st_thre',
                        help='Threshold for self training',
                        type=float,
                        default=3.0)
    parser.set_defaults(func=train)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.func(args)
