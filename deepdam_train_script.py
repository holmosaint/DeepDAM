import subprocess, os, sys, time
import numpy as np
import h5py as h5
import glob
import torch

def check_and_mkdir(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)

TOTAL_GPU = 1
MAX_JOBS = 2
def job_log(job_list):
    while len(job_list) == MAX_JOBS:
        for i, job in enumerate(job_list):
            try:
                job.wait(10)
                job_list = job_list[:i]+job_list[i+1:]
                return job_list
            except:
                continue
    return job_list

da_factor_list = [0.01, 0.001, 0.0001][2:]
da_thre_list = [0.05, 0.01, 0.001][1:2]
st_thre_list = [0.05, 0.01, 0.001][1:2]
start_idx_list = np.linspace(0, 0, 1).astype(np.int32)
min_thre_list = [0.01]
st_factor_list = [1.0]
cuda_id = 0
reps = 5
job_list = list()

# TODO: investigate batch size (both smaller)
# based on incomplete knowledge, the larger the batch size, the better
reps_st, reps_en = int(sys.argv[1]), int(sys.argv[2])
root_dir = '.'
for r in range(reps_st, reps_en):
    for t_factor in [0.1]:
        for st_factor in st_factor_list:
            for da_factor in da_factor_list:
                for da_thre in da_thre_list:
                    for st_thre in st_thre_list:
                        if st_thre > da_thre:
                            continue
                        for start_idx in start_idx_list:
                            for min_thre in min_thre_list:
                                for exp_type in ['pos_9_neg_66_20220303']:
                                    _dir = f'{root_dir}/{exp_type}/dast/min_thre_{min_thre}/t_factor_{t_factor}/{r}/da_factor_{da_factor}/da_thre_{da_thre}/st_thre_{st_thre}/start_idx_{start_idx}'
                                    os.makedirs(_dir, exist_ok=True)

                                    cmd = 'CUDA_VISIBLE_DEVICES={cuda_id} python main.py -t 100 -b 800 -seq 1 -e 100 -dir {_dir} -lr 1e-4 -size 385020 \
                                        -syn_data ./data/syn_data/data/MAT_1000_T_{t_factor}.h5 \
                                        -cnn_type resnet -cnn_config 18 -time_type lstm -out 1 -fl 0 -dynamic_key dynamic_data -target_key target_data \
                                        -exp_test_data ./data/exp_data/Hip_CA1_{exp_type}_spon.h5 -subdata 1 \
                                        -da 1 -da_factor {da_factor} -da_type CAN -da_n_kernel 5 -da_kernel_mul 2 -da_thre {da_thre} -da_eps 1e-6 -da_n_layer 1 -da_layer_offset 0 \
                                        -st 1 -st_factor {st_factor} -st_thre {st_thre} \
                                        -start_idx {start_idx} -method aug -min_thre {min_thre} -seed {seed}'.format(cuda_id=cuda_id%TOTAL_GPU, _dir=_dir, da_factor=da_factor, da_thre=da_thre, st_factor=st_factor, st_thre=st_thre, start_idx=start_idx, t_factor=round(t_factor, 2), exp_type=exp_type, min_thre=min_thre, seed=r)
                                    print(cmd)
                                    job = subprocess.Popen(cmd, shell=True)
                                    # job.wait()
                                    cuda_id += 1
                                    job_list.append(job)

                                    job_list = job_log(job_list)
for job in job_list:
    job.wait()
