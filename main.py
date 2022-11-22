import os
import pymp
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import argparse
import h5py as h5
from copy import deepcopy

from model import FeatureExtractor, Regressioner, Classifier

from DomainAdaptation.comp import DA
from SelfTraining.comp import ST

from utils import save_checkpoint, construct_dataloader, init_weights, echo, write_scalar

def construct_data(model_name, seq_num, time_len, dt, param_num, feature_num):
    if model_name == 'HH':
        from generator.HodgkinHuxley import HodgkinHuxley
        model = HodgkinHuxley(seq_num, time_len, dt, param_num, feature_num, V_0=-65)
    else:
        raise NotImplementedError('Only accept model: HH, but got {}'.format(model_name))
        
    exp_dynamic_data, exp_feature_data, feature_ratio = model.construct_exp()
        
    return model, exp_dynamic_data, exp_feature_data, feature_ratio

def construct_net(
    time_len,
    seq_num,
    output_dim,
    cnn_type,
    time_type,
    fc_layers,
    cnn_config,
    da_type,
    lr,
    store_dir,
    model_path,
    cuda=True,
    dropout=False,
):
    rank = torch.distributed.get_rank()
    feature_extractor = FeatureExtractor(
        time_len,
        seq_num,
        output_dim,
        cnn_type,
        time_type,
        fc_layers,
        cnn_config,
        cuda=cuda,
        dropout=dropout,
    )
    regressioner = Regressioner(feature_extractor.fc_dim,
                                output_dim,
                                fc_layers,
                                cuda=cuda,
                                dropout=dropout)
    exp_regressioner = deepcopy(regressioner)
    if da_type == 'DANN':
        classifier = Classifier(feature_extractor.fc_dim, fc_layers)
    else:
        classifier = nn.Identity()

    if cuda:
        feature_extractor.cuda()
        regressioner.cuda()
        exp_regressioner.cuda()
        classifier.cuda()

    feature_extractor.apply(init_weights)
    regressioner.apply(init_weights)
    feature_extractor = torch.nn.parallel.DistributedDataParallel(
        feature_extractor,
        device_ids=[rank % torch.cuda.device_count()],
        find_unused_parameters=True)
    regressioner = torch.nn.parallel.DistributedDataParallel(
        regressioner,
        device_ids=[rank % torch.cuda.device_count()],
        find_unused_parameters=True)
    opt_Adam = torch.optim.Adam(filter(
        lambda p: p.requires_grad,
        list(feature_extractor.parameters()) + list(regressioner.parameters()) +
        list(classifier.parameters())),
        lr=lr,
        betas=(0.9, 0.99))
    loss_func = nn.MSELoss()

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
        exp_params = None
        exp_score = None
        train_idx = 0
        val_idx = 0
    else:
        checkpoint = torch.load(os.path.join(model_path,
                                             'feature_extractor.h5'),
                                map_location="cuda:{}".format(rank))
        train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list = checkpoint[
            'train_loss']
        val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list = checkpoint[
            'val_loss']
        opt_Adam.load_state_dict(checkpoint['optimizer_state_dict'])
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        _epoch = checkpoint['epoch'][-1]
        try:
            last_val_loss = min(val_loss_list)
            last_val_syn_loss = min(val_syn_loss_list)
            last_val_da_loss = min(val_da_loss_list)
            last_val_st_loss = min(val_st_loss_list)
        except:
            last_val_loss = 1e8 
            last_val_syn_loss = 1e8
            last_val_da_loss = 1e8 
            last_val_st_loss = 1e8 

        train_idx = len(train_loss_list)
        val_idx = len(val_loss_list)
        exp_params, exp_score = checkpoint['exp_data']

        checkpoint = torch.load(os.path.join(model_path, 'regressioner.h5'),
                                map_location="cuda:{}".format(rank))
        regressioner.load_state_dict(checkpoint['model_state_dict'])

        if rank == 0:
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

    return feature_extractor, regressioner, classifier, opt_Adam, loss_func, writer, train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list, val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list, last_val_loss, last_val_syn_loss, last_val_da_loss, last_val_st_loss, _epoch, train_idx, val_idx, exp_params, exp_score

def simulate(simulator, params, seq_num, time_len, feature_num, normalize):
    if normalize:
        params *= simulator.param_ranges[1:] - simulator.param_ranges[:1]
        params += simulator.param_ranges[:1]
    
    dynamic = pymp.shared.array((params.shape[0], seq_num, time_len), np.float32)
    feature = pymp.shared.array((params.shape[0], feature_num), np.float32)

    with pymp.Parallel(min(30, params.shape[0])) as p:
        for i in p.range(params.shape[0]):
            dynamic[i], feature[i] = simulator.gen_single(params[i])

    return dynamic, feature

def eval_exp_data(exp_dynamic_data, exp_feature_data, simulator, feature_extractor, regressioner, batch_size, cuda):

    pred_param_arr = list() 

    for start_idx in range(0, exp_dynamic_data.shape[0], batch_size):
        dynamic_sample = torch.FloatTensor(
            exp_dynamic_data[start_idx:min(start_idx +
                                           batch_size, exp_dynamic_data.shape[0]
                                          )])
        batch = dynamic_sample.shape[0]
        if cuda:
            dynamic_sample = dynamic_sample.cuda(non_blocking=True)
        with torch.no_grad():
            prediction, _ = regressioner(feature_extractor(dynamic_sample, reset=True))

        prediction_copy = prediction.clone().detach().cpu().numpy()
        prediction_copy *= simulator.param_ranges[1:] - simulator.param_ranges[:1]
        prediction_copy += simulator.param_ranges[:1]
        pred_param_arr.append(prediction_copy)

    pred_param_arr = np.concatenate(pred_param_arr, axis=0)
    pred_dynamic_arr, pred_feature_arr = simulate(simulator, pred_param_arr, exp_dynamic_data.shape[1], exp_dynamic_data.shape[-1], exp_feature_data.shape[-1], False)
    
    return pred_dynamic_arr, pred_param_arr, pred_feature_arr


def save_model(store_dir, feature_extractor, regressioner, opt,
               pred_dynamic_arr, gt_dynamic_arr, pred_param_arr, exp_params,
               exp_score, exp_feature, train_loss_list, val_loss_list, write_model=False):
    with h5.File(os.path.join(store_dir, 'pred_exp.h5'), 'w') as f:
        f.create_dataset('pred_param', data=pred_param_arr)
        # f.create_dataset('pred_dynamic', data=pred_dynamic_arr)
        # f.create_dataset('gt_dynamic', data=gt_dynamic_arr)

    if write_model:
        save_checkpoint(store_dir, 'feature_extractor.h5', [0],
                        feature_extractor.state_dict(), opt.state_dict(),
                        train_loss_list, val_loss_list, [exp_params, exp_score, exp_feature])
        save_checkpoint(store_dir, 'regressioner.h5', [0],
                        regressioner.state_dict(), opt.state_dict(), [], [], [])

def train_step(feature_extractor,
                   regressioner,
                   dynamic_sample,
                   target_sample,
                   domain_adaptation,
                   self_training,
                   cuda,
                   loss_func,
                   opt=None):
    # Synthesized Data Training
    batch_size = dynamic_sample.shape[0]
    if cuda:
        dynamic_sample = dynamic_sample.cuda(non_blocking=True)
        target_sample = target_sample.cuda(non_blocking=True)
    syn_feature = feature_extractor(dynamic_sample, reset=True)
    syn_prediction, syn_fc_feature = regressioner(syn_feature)
    syn_loss = loss_func(syn_prediction, target_sample)
    if opt is not None:
        opt.zero_grad()
        syn_loss.backward()
        opt.step()

    # Self Training
    st_dynamic_sample, st_target_sample = self_training.sample()
    st_prediction = list()
    st_loss = torch.zeros((1)).to(device=dynamic_sample.device)
    if self_training.active:
        if cuda:
            st_dynamic_sample = st_dynamic_sample.cuda(non_blocking=True)
            st_target_sample = st_target_sample.cuda(non_blocking=True)
        for i in range(0, st_dynamic_sample.shape[0], batch_size):
            _end = min(i+batch_size, st_dynamic_sample.shape[0])
            feature = feature_extractor(st_dynamic_sample[i:_end], reset=True)
            prediction, fc_feature = regressioner(feature)
            loss = loss_func(prediction, st_target_sample[i:_end])
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            st_loss += loss * (_end - i)
            st_prediction.append(prediction.detach().cpu().numpy())
        st_loss /= st_dynamic_sample.shape[0]
        if len(st_prediction) > 0:
            st_prediction = np.concatenate(st_prediction, axis=0)

    # Domain Adaptation Training
    da_S_dynamic_sample, da_T_dynamic_sample = self_training.sample_pair(20)
    da_loss = torch.zeros((1)).to(device=dynamic_sample.device)
    if domain_adaptation.active:
        if cuda:
            da_S_dynamic_sample = da_S_dynamic_sample.cuda(non_blocking=True)
            da_T_dynamic_sample = da_T_dynamic_sample.cuda(non_blocking=True)
        for i in range(da_S_dynamic_sample.shape[0]):
            da_dynamic = torch.cat([da_S_dynamic_sample[i:i+1], da_T_dynamic_sample[i]], dim=0)
            da_feature_lstm = feature_extractor(da_dynamic, reset=True)
            _, da_feature_fc = regressioner(da_feature_lstm, _end=domain_adaptation.layer_offset)
            da_S_feature = [da_feature_lstm[:1]] + [da_feature_fc[ii][:1] for ii in range(len(da_feature_fc))]
            da_T_feature = [da_feature_lstm[1:]] + [da_feature_fc[ii][1:] for ii in range(len(da_feature_fc))]
            loss = domain_adaptation.loss(da_S_feature, da_T_feature)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            da_loss += loss
        da_loss /= da_S_dynamic_sample.shape[0]
    
    loss = syn_loss + da_loss + st_loss
    return loss, syn_loss, da_loss, st_loss, opt

def train(args):
    rank = dist.get_rank()
    
    time_len = args.t
    dt = args.dt
    batch_size = args.b
    epoch = args.e
    cuda = args.c
    store_dir = args.dir
    lr = args.lr
    data_path = args.data
    model_name = args.syn_model
    seq_num = args.seq
    model_path = args.model
    output_dim = args.out
    feature_num = args.fdim
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

    echo(batch_size)

    feature_extractor, regressioner, da_classifier, opt_Adam, loss_func, writer, train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list, val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list, last_val_loss, last_val_syn_loss, last_val_da_loss, last_val_st_loss, _epoch, train_idx, val_idx, exp_params, exp_score = construct_net(
        time_len,
        seq_num,
        output_dim,
        cnn_type,
        time_type,
        fc_layers,
        cnn_config,
        args.da_type,
        lr,
        store_dir,
        model_path,
        cuda=cuda,
        dropout=dropout,
    )

    train_loader, val_loader = construct_dataloader(data_size, data_path,
                                                    dynamic_key, feature_key,
                                                    target_key, batch_size,
                                                    subdata)
    syn_model, exp_dynamic_data, exp_feature_data, feature_ratio = construct_data(model_name, seq_num, time_len, dt, output_dim, feature_num)

    not_improve = torch.FloatTensor([0])
    if cuda:
        not_improve = not_improve.cuda()

    # Domain Adaptation
    domain_adaptation = DA(args.da, args.da_type, args.da_layer,
                           args.da_n_layer, args.da_layer_offset,
                           args.da_n_kernel, args.da_factor, da_classifier)

    # Self Training
    self_training = ST(args.st, args.st_factor, exp_dynamic_data,
                       exp_feature_data, feature_ratio, syn_model.param_ranges, args.st_thre, loss_func,
                       args.seq, args.st_bank_size, exp_params, exp_score)
    if exp_params is not None:
        self_training.syn_dynamic_data, _ = simulate(syn_model, exp_params, seq_num, time_len, feature_num, False)

    pred_dynamic_data, pred_param_data, pred_feature_data = eval_exp_data(exp_dynamic_data, exp_feature_data, syn_model, feature_extractor, regressioner, batch_size, cuda)
    self_training.update(pred_param_data, pred_dynamic_data, pred_feature_arr=pred_feature_data, cuda=cuda)

    last_best_score = np.ones(exp_dynamic_data.shape[0]) * 1e8
    not_improve = 0
    is_break = False
    for e in range(_epoch, epoch):
        if is_break:
            break
        for batch_idx, (dynamic_sample, feature_sample,
                        target_sample) in enumerate(train_loader):
            print("Train Batch: ", batch_idx, end='\r')

            total_good_num, cur_good_num, good_MAE, update_num = self_training.update(target_sample.numpy() * (syn_model.param_ranges[1:] - syn_model.param_ranges[:1]) + syn_model.param_ranges[:1], dynamic_sample.numpy(), pred_feature_arr=feature_sample.numpy(), cuda=cuda)

            feature_extractor.train()
            regressioner.train()
            loss, syn_loss, da_loss, st_loss, opt_Adam = train_step(feature_extractor, regressioner, dynamic_sample, target_sample, domain_adaptation, self_training, cuda, loss_func, opt=opt_Adam)

            # Experiment Evaluation
            feature_extractor.eval()
            regressioner.eval()
            pred_dynamic_data, pred_param_data, pred_feature_data = eval_exp_data(exp_dynamic_data, exp_feature_data, syn_model, feature_extractor, regressioner, dynamic_sample.shape[0], cuda)

            # Update Self Training
            total_good_num, cur_good_num, good_MAE, update_num = self_training.update(
                pred_param_data, pred_dynamic_data, pred_feature_arr=pred_feature_data, cuda=cuda)

            # Log
            train_loss_list.append(loss.item())
            train_syn_loss_list.append(syn_loss.item())
            train_da_loss_list.append(da_loss.item())
            train_st_loss_list.append(st_loss.item())

            cur_best_score = np.min(self_training.exp_score, axis=-1)
            update = np.argwhere(cur_best_score < last_best_score).shape[0] > 0
            last_best_score = np.min(np.concatenate([cur_best_score[:, None], last_best_score[:, None]], axis=-1), axis=-1)

            if rank == 0:
                write_scalar(writer, 'Training Loss', loss.item(), train_idx)
                write_scalar(writer, 'Training Syn Loss', syn_loss.item(),
                             train_idx)
                write_scalar(writer, 'Training DA Loss', da_loss.item(),
                             train_idx)
                write_scalar(writer, 'Training ST Loss', st_loss.item(),
                             train_idx)
                write_scalar(writer, '# of Total Exp Train', total_good_num,
                             train_idx)
                write_scalar(writer, '# of Epoch Exp Good', cur_good_num,
                             train_idx)
                write_scalar(writer, 'Good Feature MAE', good_MAE, train_idx)
                write_scalar(writer, '# of Update', update_num, train_idx)
                write_scalar(writer, 'Not Improve', not_improve, train_idx)
                write_scalar(writer, 'Exp Train Ratio',
                             total_good_num / exp_dynamic_data.shape[0], train_idx)
                if model_name == 'HH':
                    for thre in np.linspace(0.5, 2.0, 7):
                        write_scalar(writer, 'Thre {}'.format(thre),
                                    self_training.cal_good(self_training.exp_score[:, 0], thre=thre).shape[0], train_idx)
                if writer is not None:
                    try:
                        writer.add_histogram(
                            'Feature MIN MAE Hist',
                            self_training.exp_score[:, 0])

                        writer.add_histogram(
                            'Feature MAX MAE Hist',
                            self_training.exp_score[:, -1])
                    except:
                        pass
                    writer.flush()
                if update:
                    save_model(
                        store_dir, feature_extractor, regressioner, opt_Adam,
                        self_training.syn_dynamic_data, exp_dynamic_data, pred_param_data,
                        self_training.exp_params, self_training.exp_score, self_training.exp_feature, [
                            train_loss_list, train_syn_loss_list,
                            train_da_loss_list, train_st_loss_list,
                        ], [
                            val_loss_list, val_syn_loss_list, val_da_loss_list,
                            val_st_loss_list
                        ], write_model=True if train_idx % 100 == 0 else False)
                    not_improve = 0
                else:
                    not_improve += 1
                    if not_improve % 1000 == 0:
                        for p in opt_Adam.param_groups:
                            p['lr'] /= 2
                    if not_improve % 100000 == 0:
                        is_break = True
                        break

            train_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Architecture arguments
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('-t', help='Time series length', default=5000, type=int)
    parser.add_argument('-dt', help='Time step for simulation', default=0.1, type=float)
    parser.add_argument('-syn_model', help='Name of synthesized model', type=str, required=True)
    parser.add_argument('-b', help='Batch size', default=32, type=int)
    parser.add_argument('-e', help='Epoch', default=100, type=int)
    parser.add_argument('-c', help='Cuda', default=True, type=bool)
    parser.add_argument('-dir', help='Store directory', default='.', type=str)
    parser.add_argument('-lr', help='Learning rate', default=1e-3, type=float)
    parser.add_argument('-data',
                        help='Path to data .h5 file',
                        required=True,
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
    parser.add_argument('-fdim', help='# of features to calculate similarity', default=0, type=int)
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

    # Domain Adaptation
    parser.add_argument('-da',
                        help='Whether to use domain adaptation',
                        type=int,
                        required=True)
    parser.add_argument(
        '-da_type',
        help=
        'Type of domain adaptation techniques to use: [DDC, DAN, CORAL, CDM, DANN]',
        type=str,
        default='DDC')
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
    parser.add_argument('-da_layer_offset',
                        help='Layer offset to calculate da loss, > 1',
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
    parser.add_argument('-st_bank_size',
                        help='Bank size for self training',
                        type=int,
                        default=1)
    parser.set_defaults(func=train)

    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.func(args)
