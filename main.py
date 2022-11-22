import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from sklearn.metrics import matthews_corrcoef
from math import ceil

matplotlib.use('Agg')
import argparse
import h5py as h5

from model import FeatureExtractor, Regressioner

from DomainAdaptation.comp import DA
from SelfTraining.comp import ST

from utils import save_checkpoint, construct_dataloader, init_weights, echo, write_scalar, ExpDataset


def construct_exp(exp_test_data_path, batch_size):

    with h5.File(exp_test_data_path, 'r') as f:
        exp_dynamic_data_test = f.get('dynamic_data')[...].astype(np.float32)
        exp_target_data_test = f.get('target_data')[...].reshape(-1)
        exp_target_data_test[exp_target_data_test != 0] = 1
        exp_target_data_test = exp_target_data_test.astype(np.int32)
    exp_dataset = ExpDataset(exp_test_data_path, 'dynamic_data', 'target_data', 0, exp_dynamic_data_test.shape[0])
    exp_sampler = torch.utils.data.distributed.DistributedSampler(exp_dataset, shuffle=True)
    exp_dataloader = torch.utils.data.DataLoader(exp_dataset, batch_size=batch_size, sampler=exp_sampler, num_workers=1)
       
    return exp_dynamic_data_test, exp_target_data_test, exp_dataloader

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
    rank = torch.distributed.get_rank()
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
    regressioner = Regressioner(feature_extractor.fc_dim,
                                output_dim,
                                fc_layers,
                                cuda=cuda,
                                dropout=dropout)

    if cuda:
        feature_extractor.cuda()
        regressioner.cuda()

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
        list(feature_extractor.parameters()) + list(regressioner.parameters())),
                                lr=lr,
                                betas=(0.9, 0.99))
    loss_func = nn.BCEWithLogitsLoss()

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
        _epoch = 0
        train_idx = 0
        val_idx = 0
    else:
        checkpoint = torch.load(os.path.join(model_path,
                                             'feature_extractor.h5'),
                                map_location="cuda:{}".format(rank))
        train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list = checkpoint['train_loss']
        val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list = checkpoint['val_loss']
        opt_Adam.load_state_dict(checkpoint['optimizer_state_dict'])
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        _epoch = checkpoint['epoch'][-1]
        last_val_loss = min(val_loss_list)
        train_idx = len(train_loss_list)
        val_idx = len(val_loss_list)

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

    return feature_extractor, regressioner, opt_Adam, loss_func, writer, train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list, val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list, last_val_loss, _epoch, train_idx, val_idx


def eval_exp_data(exp_dynamic_data, feature_extractor, regressioner, batch_size, cuda):

    pred_param_arr = list()
    for start_idx in range(0, exp_dynamic_data.shape[0], batch_size):
        dynamic_sample = torch.FloatTensor(
            exp_dynamic_data[start_idx:min(start_idx + batch_size, exp_dynamic_data.shape[0])])
        batch = dynamic_sample.shape[0]
        domain_sample = torch.ones(batch).to(torch.long)
        if cuda:
            dynamic_sample = dynamic_sample.cuda(non_blocking=True)
            domain_sample = domain_sample.cuda(non_blocking=True)
        with torch.no_grad():
            prediction, _ = regressioner(feature_extractor(dynamic_sample, domain_sample, reset=True))
            prediction = torch.sigmoid(prediction)

        pred_param_arr.append(prediction.detach().cpu().numpy())

    pred_param_arr = np.concatenate(pred_param_arr, axis=0)
    pred_score_arr = np.copy(pred_param_arr[:, 0])
    pred_param_arr[pred_param_arr < 0.5] = 0
    pred_param_arr[pred_param_arr >= 0.5] = 1
    pred_param_arr = pred_param_arr.reshape(-1)

    return pred_param_arr, pred_score_arr


def save_model(store_dir, feature_extractor, regressioner, opt, train_loss_list, val_loss_list, exp_dynamic_data_unknown, pred_param_data, pred_score_data):

    save_checkpoint(store_dir, 'feature_extractor.h5', [0],
                    feature_extractor.state_dict(), opt.state_dict(),
                    train_loss_list, val_loss_list,
                    [exp_dynamic_data_unknown, pred_param_data, pred_score_data])
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
    st_dynamic_sample, st_target_sample = self_training.sample(batch_size)
    st_loss = torch.zeros([1]).to(device=syn_loss.device)
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
    if st_dynamic_sample.shape[0] > 0:
        st_loss /= st_dynamic_sample.shape[0]

    # Domain Adaptation Training
    if domain_adaptation.active:
        syn_dynamic_sample_pos, syn_target_sample_pos, syn_dynamic_sample_neg, syn_target_sample_neg, exp_dynamic_sample_pos, exp_target_sample_pos, exp_dynamic_sample_neg, exp_target_sample_neg, st_mask = domain_adaptation.sample(feature_extractor, regressioner)
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
            syn_step_pos = np.linspace(0, syn_size_pos, num=n_split+1).astype(np.int)
            syn_step_neg = np.linspace(0, syn_size_neg, num=n_split+1).astype(np.int)
            exp_step_pos = np.linspace(0, exp_size_pos, num=n_split+1).astype(np.int)
            exp_step_neg = np.linspace(0, exp_size_neg, num=n_split+1).astype(np.int)

            for i in range(n_split):
                syn_sample_pos = syn_dynamic_sample_pos[syn_step_pos[i]:syn_step_pos[i+1]]
                syn_sample_neg = syn_dynamic_sample_neg[syn_step_neg[i]:syn_step_neg[i+1]]
                exp_sample_pos = exp_dynamic_sample_pos[exp_step_pos[i]:exp_step_pos[i+1]]
                exp_sample_neg = exp_dynamic_sample_neg[exp_step_neg[i]:exp_step_neg[i+1]]
                _syn_size = syn_sample_pos.shape[0] + syn_sample_neg.shape[0]
                _exp_size = exp_sample_pos.shape[0] + exp_sample_neg.shape[0]

                da_dynamic = torch.cat([syn_sample_pos, syn_sample_neg, exp_sample_pos, exp_sample_neg], dim=0)

                da_feature = feature_extractor(da_dynamic, reset=True)
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
    rank = torch.distributed.get_rank()
    num_proc = torch.distributed.get_world_size()

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
    target_key = args.target_key
    min_thre = args.min_thre

    echo(batch_size)

    feature_extractor, regressioner, opt_Adam, loss_func, writer, train_loss_list, train_syn_loss_list, train_da_loss_list, train_st_loss_list, val_loss_list, val_syn_loss_list, val_da_loss_list, val_st_loss_list, last_val_loss, _epoch, train_idx, val_idx = construct_net(
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
                                                    dynamic_key, 
                                                    target_key, batch_size, min_thre
                                                    )
    exp_dynamic_data_unknown, exp_target_data_unknown, exp_dataloader = construct_exp(exp_test_data_path, batch_size)

    not_improve = torch.FloatTensor([0])
    if cuda:
        not_improve = not_improve.cuda()

    # Domain Adaptation
    domain_adaptation = DA(args.da, args.da_type, args.da_layer,
                           args.da_n_layer, args.da_layer_offset,
                           args.da_n_kernel, args.da_kernel_mul, args.da_factor, train_loader, exp_dataloader, args.da_thre, args.st_thre, args.da_eps, args.da_intra_only)

    # Self Training
    self_training = ST(args.st, args.st_factor, exp_dynamic_data_unknown, exp_target_data_unknown, args.st_thre)

    is_break = False
    for e in range(_epoch, epoch):
        if is_break:
            break

        for batch_idx, (dynamic_sample, target_sample) in enumerate(train_loader):
            print("Train Batch: ", batch_idx, end='\r')

            feature_extractor.train()
            regressioner.train()

            loss, syn_loss, da_loss, st_loss, opt_Adam, exp_da_data = train_step(feature_extractor, regressioner, dynamic_sample, target_sample, domain_adaptation, self_training, cuda, loss_func, opt=opt_Adam)

            # Experiment Evaluation
            feature_extractor.eval()
            regressioner.eval()

            pred_param_data, pred_score_data = eval_exp_data(exp_dynamic_data_unknown, feature_extractor, regressioner, dynamic_sample.shape[0], cuda)

            # Training Log
            train_loss_list.append(loss.item())
            train_syn_loss_list.append(syn_loss.item())
            train_da_loss_list.append(da_loss.item())
            train_st_loss_list.append(st_loss.item())

            # Write Training Log
            if rank == 0:
                write_scalar(writer, 'Training Loss', loss.item(), train_idx)
                write_scalar(writer, 'Training Syn Loss', syn_loss.item(), train_idx)
                write_scalar(writer, 'Training DA Loss', da_loss.item(), train_idx)
                write_scalar(writer, 'Training ST Loss', st_loss.item(), train_idx)
                print(pred_param_data)
                pred_param_data[pred_param_data > 0.5] = 1
                pred_param_data[pred_param_data <= 0.5] = 0
                write_scalar(writer, 'Epoch Acc', np.argwhere(pred_param_data == exp_target_data_unknown).shape[0] / pred_param_data.shape[0], train_idx)
                write_scalar(writer, 'Epoch TP',np.argwhere(np.logical_and(pred_param_data == exp_target_data_unknown, exp_target_data_unknown == 1)).shape[0] / np.argwhere(exp_target_data_unknown == 1).shape[0], train_idx)
                write_scalar(writer, 'Epoch TN',np.argwhere(np.logical_and(pred_param_data == exp_target_data_unknown, exp_target_data_unknown == 0)).shape[0] / np.argwhere(exp_target_data_unknown == 0).shape[0], train_idx)
                write_scalar(writer, 'Epoch FP',np.argwhere(np.logical_and(pred_param_data != exp_target_data_unknown, exp_target_data_unknown == 0)).shape[0] / np.argwhere(exp_target_data_unknown == 0).shape[0], train_idx)
                write_scalar(writer, 'Epoch FN',np.argwhere(np.logical_and(pred_param_data != exp_target_data_unknown, exp_target_data_unknown == 1)).shape[0] / np.argwhere(exp_target_data_unknown == 1).shape[0], train_idx)

                cur_mcc = matthews_corrcoef(exp_target_data_unknown, pred_param_data)
                write_scalar(writer, 'Epoch MCC', cur_mcc, train_idx)
                write_scalar(writer, 'Pos pred size', np.argwhere(pred_param_data == 1).shape[0], train_idx)

            # Use Synthesized Data for Validation
            feature_extractor.eval()
            regressioner.eval()
            tmp_loss_list = list()
            tmp_syn_loss_list = list()
            tmp_da_loss_list = list()
            tmp_st_loss_list = list()
            max_iter = 10
            _iter = 0
            with torch.no_grad():
                for (dynamic_sample, target_sample) in val_loader:
                    _iter += 1
                    if _iter >= max_iter:
                        break
                    loss, syn_loss, da_loss, st_loss, _, _ = train_step(feature_extractor, regressioner, dynamic_sample, target_sample, domain_adaptation, self_training, cuda, loss_func, opt=None)

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
            torch.distributed.reduce(val_loss,
                                        0,
                                        op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(val_syn_loss,
                                        0,
                                        op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(val_da_loss,
                                        0,
                                        op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(val_st_loss,
                                        0,
                                        op=torch.distributed.ReduceOp.SUM)
            val_loss /= num_proc
            val_syn_loss /= num_proc
            val_da_loss /= num_proc
            val_st_loss /= num_proc
            val_loss_list.append(val_loss)
            val_syn_loss_list.append(val_syn_loss)
            val_da_loss_list.append(val_da_loss)
            val_st_loss_list.append(val_st_loss)
            if rank == 0:
                write_scalar(writer, 'Val Loss', val_loss, val_idx)
                write_scalar(writer, 'Val Syn Loss', val_syn_loss, val_idx)
                write_scalar(writer, 'Val DA Loss', val_da_loss, val_idx)
                write_scalar(writer, 'Val ST Loss', val_st_loss, val_idx)
            val_idx += 1
            
            update = loss.item() < last_val_loss
            last_val_loss = min(last_val_loss, loss.item())

            # Update Self Training
            self_training.update(exp_da_data[0][exp_da_data[2]], exp_da_data[1][exp_da_data[2]], update=update)

            if not update:
                not_improve += 1
                if not_improve % 10 == 0:
                    for p in opt_Adam.param_groups:
                        p['lr'] /= 2
                if not_improve % 50 == 0:
                    is_break = True
                    break
            else:
                not_improve = 0
                with h5.File(os.path.join(store_dir, 'pred.h5'), 'w') as f:
                    f.create_dataset('pred_param', data=pred_score_data)
                    f.create_dataset('target_param', data=exp_target_data_unknown)

            write_scalar(writer, 'Update', int(update), train_idx)
            write_scalar(writer, 'Not Improve', not_improve, train_idx)

            if update:
                save_model(
                    store_dir, feature_extractor, regressioner, opt_Adam,
                    exp_dynamic_data_unknown, pred_param_data, pred_score_data, 
                    [
                        train_loss_list, train_syn_loss_list,
                        train_da_loss_list, train_st_loss_list,
                    ], 
                    [
                        val_loss_list, val_syn_loss_list, val_da_loss_list,
                        val_st_loss_list,
                    ])

            train_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Architecture arguments
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
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
    parser.add_argument('-fl', help='FC layers', default=0, type=int)
    parser.add_argument('-dr', help='Dropout', default=0, type=int)
    parser.add_argument('-dynamic_key',
                        help='dynamic key of the data file',
                        default=None,
                        type=str)
    parser.add_argument('-target_key',
                        help='target key of the data file',
                        default=None,
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
                        help='# of kernels to use in CAN, > 1',
                        type=int,
                        default=2)
    parser.add_argument('-da_kernel_mul',
                        help='Sigma base for kernels to use in CAN, > 1',
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
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    args.func(args)
