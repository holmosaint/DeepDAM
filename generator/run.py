from HodgkinHuxley import HodgkinHuxley
from STG import STG
import argparse

from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np


def gen_data(args):
    time_len = args.t
    dt = args.d
    num = args.num
    seq_num = args.seq
    param_num = args.param_num
    feature_num = args.feature_num
    store_file = args.store_file
    model_name = args.neural_model
    NUM_THREAD = args.threads

    if model_name == 'HH':
        neural_model = HodgkinHuxley(seq_num,
                                     time_len,
                                     dt,
                                     param_num,
                                     feature_num,
                                     V_0=-75)
    elif model_name == 'stg':
        neural_model = STG(seq_num,
                           time_len,
                           dt,
                           param_num,
                           feature_num)
    else:
        raise NotImplementedError

    neural_model.construct_pilot(num, store_file, NUM_THREAD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(description="Command Parser: [data]")

    data_generator_parser = subparser.add_parser('data')
    data_generator_parser.add_argument('-t',
                                       help='Time series length',
                                       default=2000,
                                       type=int)
    data_generator_parser.add_argument('-d',
                                       help='Time step length',
                                       default=0.1,
                                       type=float)
    data_generator_parser.add_argument('-num',
                                       help='num',
                                       default=100,
                                       type=int)
    data_generator_parser.add_argument('-seq',
                                       help='Number of Sequences per Neuron',
                                       default=20,
                                       type=int)
    data_generator_parser.add_argument('-param_num',
                                       help='Number of Params per Neuron',
                                       required=True,
                                       type=int)
    data_generator_parser.add_argument('-feature_num',
                                       help='Number of Features for Similarity',
                                       required=True,
                                       type=int)
    data_generator_parser.add_argument('-threads',
                                       help='Number of Threads',
                                       default=30,
                                       type=int)
    data_generator_parser.add_argument('-store_file',
                                       help='Store file',
                                       default='./data.h5',
                                       type=str)
    data_generator_parser.add_argument(
        '-neural_model',
        help='Name of neural model: [HH]',
        required=True,
        type=str)
    data_generator_parser.set_defaults(func=gen_data)

    args = parser.parse_args()
    args.func(args)
