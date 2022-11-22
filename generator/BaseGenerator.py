import os
import numpy as np
import pymp
from tqdm import tqdm
import argparse
from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    def __init__(self, seq_num, time_len, dt, param_num, feature_num):
        super(BaseGenerator, self).__init__()
        self.seq_num = seq_num
        self.time_len = time_len
        self.dt = dt
        self.param_num = param_num
        self.feature_num = feature_num

    @abstractmethod
    def construct_exp(self):
        pass

    @abstractmethod
    def construct_pilot(self, num, store_file, NUM_THREAD=1):
        pass

    @abstractmethod
    def gen_single(self, stimulus, params):
        pass

    def gen(self,
            num,
            params,
            NUM_THREAD=1,
            vis=True):

        assert len(params.shape) == 2
        assert params.shape[0] == num

        dynamic_data = pymp.shared.array([num, self.seq_num, self.time_len], dtype='float32')
        feature_data = pymp.shared.array([num, self.feature_num], dtype='float32')
        with pymp.Parallel(NUM_THREAD) as p:
            for e in tqdm(p.range(num)) if p.thread_num == 0 and vis else p.range(num):
                dynamic_data[e], feature_data[e] = self.gen_single(params[e])

        return dynamic_data, feature_data
