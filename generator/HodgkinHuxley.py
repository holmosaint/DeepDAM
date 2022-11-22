try:
    from .BaseGenerator import BaseGenerator
except:
    from BaseGenerator import BaseGenerator
import numpy as np
import h5py as h5
from numba import njit
import efel

from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import warnings

warnings.filterwarnings('ignore')

def extract_features(dynamic_data):
    assert dynamic_data.shape[0] == 1, 'Currently only support seq_num==1'
    amp = 0.13
    exc_val = np.nan
    feature_list = list()
    I = np.ones(7000) * amp * 1e3
    I[:600] = 0
    I[-1400:] = 0
    sweep_ext = EphysSweepFeatureExtractor(
        t=np.linspace(0, 7000 * 0.2, 7000) * 1e-3,
        v=dynamic_data[0],
        i=I,
        start=0.12,
        end=1.12,
        filter=2.49)
    sweep_ext.process_spikes()

    feature_names = [
        'peak_v',
        'fast_trough_v',
        'slow_trough_v',
        'slow_trough_t',
        'width',
        'peak_t',
        'adp_index',
    ]
    fea_dict = dict()
    for key in feature_names:
        fea_dict[key] = sweep_ext.spike_feature(key)
    firing_rate = np.array([
        len(fea_dict['peak_v'])
    ]) if len(fea_dict['peak_v']) > 0 else np.array([0])
    ap_peak = [np.nanmean(fea_dict['peak_v'])]
    fast_trough_v = [np.nanmean(fea_dict['fast_trough_v'])]
    slow_trough_v = [np.nanmean(fea_dict['slow_trough_v'])]
    slow_trough_t = [np.nanmean(fea_dict['slow_trough_t'])]
    width = [np.nanmean(fea_dict['width'])]
    t2firstspike = np.array([fea_dict['peak_t'][0]]) - 0.12 if len(
        fea_dict['peak_t']) > 0 else np.array([np.nan])
    if len(fea_dict['peak_t']) >= 2:
        ISI = np.array(fea_dict['peak_t'])[1:] - np.array(
            fea_dict['peak_t'])[:-1]
        adp = [np.mean((ISI[1:] - ISI[:-1]) / (ISI[1:] + ISI[:-1]))]
    else:
        ISI = np.array([np.nan])
        adp = np.array([np.nan])

    ISI_first = [ISI[0]]
    ISI_cv = [np.std(ISI) / np.mean(ISI)]
    ISI_mean = [np.mean(ISI)]
    # adp = [np.nanmean(fea_dict['adp_index'])]

    feature_name = ['voltage_base',
                    'steady_state_voltage_stimend']  # 'time_constant'
    trace = [{
        'stim_start': [120],
        'stim_end': [1120],
        'T': np.linspace(0, 7000 * 0.2, 7000),
        'V': dynamic_data[0]
    }]
    feature = efel.getFeatureValues(trace,
                                    feature_name,
                                    raise_warnings=False)
    voltage_base = np.array([feature[i]['voltage_base'][0] \
            if feature[i]['voltage_base'].shape[0] > 0 else exc_val \
            for i in range(len(feature))])
    spon_fire = [
        int(
            np.argwhere(dynamic_data[0, :600] > -40).shape[0] > 0 or
            np.argwhere(dynamic_data[0, 5700:] > -40).shape[0] > 0)
    ]

    feature = np.concatenate([
        firing_rate, ap_peak, fast_trough_v, slow_trough_v,
        slow_trough_t, width, t2firstspike, ISI_first, ISI_cv, ISI_mean,
        adp, voltage_base, spon_fire
    ])
    feature[np.isnan(feature)] = 0
    feature_list.append(feature)

    return np.concatenate(feature_list)

@njit
def gen_single(stimulus, params, dt, V_0, noise):
    param_num = params.shape[0]
    assert param_num in [3, 5, 8]
    """ 3 Params Model """
    gbar_Na = params[0]  # mS/cm2
    gbar_K = params[1]  # mS/cm2
    gbar_leak = params[2]  # mS/cm2

    if param_num >= 5:
        """ 5 Params Model """
        gbar_M = params[3]  # mS/cm2
        tau_max = params[4]  # ms
    else:
        gbar_M = 0.07  # mS/cm2
        tau_max = 6e2  # ms

    if param_num >= 7:
        """ 7 Params Model """
        Vt = params[5]  # mV
        nois_fact = params[6]  # uA/cm2
        E_leak = params[7]  # mV
    else:
        Vt = -60.  # mV
        nois_fact = 0.1  # uA/cm2
        E_leak = -70.  # mV

    # fixed parameters
    C = 1.  # uF/cm2
    E_Na = 53  # mV
    E_K = -107  # mV
    V0 = V_0  # mV

    tstep = dt

    ####################################
    # kinetics
    def efun(z):
        if np.abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (np.exp(z) - 1)

    def alpha_m(x):
        v1 = x - Vt - 13.
        return 0.32 * efun(-0.25 * v1) / 0.25

    def beta_m(x):
        v1 = x - Vt - 40
        return 0.28 * efun(0.2 * v1) / 0.2

    def alpha_h(x):
        v1 = x - Vt - 17.
        return 0.128 * np.exp(-v1 / 18.)

    def beta_h(x):
        v1 = x - Vt - 40.
        return 4.0 / (1 + np.exp(-0.2 * v1))

    def alpha_n(x):
        v1 = x - Vt - 15.
        return 0.032 * efun(-0.2 * v1) / 0.2

    def beta_n(x):
        v1 = x - Vt - 10.
        return 0.5 * np.exp(-v1 / 40)

    # steady-states and time constants
    def tau_n(x):
        return 1 / (alpha_n(x) + beta_n(x))

    def n_inf(x):
        return alpha_n(x) / (alpha_n(x) + beta_n(x))

    def tau_m(x):
        return 1 / (alpha_m(x) + beta_m(x))

    def m_inf(x):
        return alpha_m(x) / (alpha_m(x) + beta_m(x))

    def tau_h(x):
        return 1 / (alpha_h(x) + beta_h(x))

    def h_inf(x):
        return alpha_h(x) / (alpha_h(x) + beta_h(x))

    # slow non-inactivating K+
    def p_inf(x):
        v1 = x + 35.
        return 1.0 / (1. + np.exp(-0.1 * v1))

    def tau_p(x):
        v1 = x + 35.
        return tau_max / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1))

    time_len = stimulus.shape[-1]
    ####################################
    # simulation from initial point
    V = np.zeros(time_len)  # voltage
    n = np.zeros(time_len)
    m = np.zeros(time_len)
    h = np.zeros(time_len)
    p = np.zeros(time_len)

    V[0] = float(V0)
    n[0] = n_inf(V[0])
    m[0] = m_inf(V[0])
    h[0] = h_inf(V[0])
    p[0] = p_inf(V[0])

    for i in range(1, time_len):
        tau_V_inv = ((m[i - 1]**3) * gbar_Na * h[i - 1] +
                     (n[i - 1]**4) * gbar_K + gbar_leak + gbar_M * p[i - 1]) / C
        V_inf = (
            (m[i - 1]**3) * gbar_Na * h[i - 1] * E_Na +
            (n[i - 1]**4) * gbar_K * E_K + gbar_leak * E_leak +
            gbar_M * p[i - 1] * E_K + stimulus[i - 1] + nois_fact * noise[i] /
            (tstep**0.5)) / (tau_V_inv * C)
        V[i] = V_inf + (V[i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
        n[i] = n_inf(
            V[i]) + (n[i - 1] - n_inf(V[i])) * np.exp(-tstep / tau_n(V[i]))
        m[i] = m_inf(
            V[i]) + (m[i - 1] - m_inf(V[i])) * np.exp(-tstep / tau_m(V[i]))
        h[i] = h_inf(
            V[i]) + (h[i - 1] - h_inf(V[i])) * np.exp(-tstep / tau_h(V[i]))
        p[i] = p_inf(
            V[i]) + (p[i - 1] - p_inf(V[i])) * np.exp(-tstep / tau_p(V[i]))

    return V

class HodgkinHuxley(BaseGenerator):

    def __init__(self,
                 seq_num,
                 time_len,
                 dt,
                 param_num,
                 feature_num,
                 V_0=-70):
        super().__init__(seq_num, time_len, dt, param_num, feature_num)
        prior_min = np.array([[0.5, 1e-4, 1e-4, 1e-4, 50,   -90,  1e-4, -100]])
        prior_max = np.array([[80., 15,   0.6,  0.6,  3000, -40., 0.15, -30]])
        self.param_ranges = np.concatenate([prior_min, prior_max], axis=0)
        self.V_0 = V_0

        amp = 0.13        
        stimulus = np.zeros((seq_num, time_len))
        stimulus[0, 600:5600] = amp * 1e-3 / (0.0153 / 184)
        self.set_stimulus(stimulus)

    def construct_exp(self):
        with h5.File('/home/shengkaiwen/Data/HH/AllenData_130pA.h5', 'r') as f:
            exp_dynamic_data = f.get('dynamic_data')[:]
            if len(exp_dynamic_data.shape) == 2:
                exp_dynamic_data = exp_dynamic_data[:, None]
            exp_feature_data = f.get('feature_data')[:]
            exp_feature_data[exp_feature_data == -1e8] = 0
        feature_ratio = np.array([0.5, 2, 2, 2, 5e-2, 1e-4, 5e-3, 1e-3, 1e-2, 5e-4, 1e-3, 2, 100])
        return exp_dynamic_data, exp_feature_data, feature_ratio

    def gen_single(self, params, V_0=None):
        stimulus = self.get_stimulus()
        rng = np.random.RandomState()

        rest_len = 1000
        noise = rng.randn(self.time_len + rest_len)
        stimulus = np.concatenate([np.zeros((self.seq_num, rest_len), np.float32), stimulus], axis=1)
        V_0 = self.V_0 if V_0 is None else V_0
        dynamic_data = np.zeros((self.seq_num, self.time_len))
        feature_data = np.zeros((self.feature_num))
        for i in range(self.seq_num):
            dynamic_data[i] = gen_single(stimulus[i], params, self.dt, V_0, noise)[rest_len:].astype(np.float32)
        feature_data = extract_features(dynamic_data)
        return dynamic_data, feature_data

    def set_stimulus(self, stimulus=None):
        self.stimulus = stimulus

    def get_stimulus(self):
        return self.stimulus

    def construct_pilot(self, num, store_file, NUM_THREAD=1):
        cnt = 0
        param_arr = np.zeros((num, self.param_num), dtype=np.float32)
        dynamic_arr = np.zeros((num, self.seq_num, self.time_len), dtype=np.float32)
        feature_arr = np.zeros((num, self.feature_num), dtype=np.float32)

        while cnt < num:
            param = np.random.uniform(self.param_ranges[0], self.param_ranges[1], size=(num, self.param_num)).astype(np.float32)

            dynamic, feature = self.gen(num-cnt, param, NUM_THREAD)
            good_idx = np.arange(dynamic.shape[0]).astype(np.int32)

            # for data wash
            if True:
                spike_idx = np.argwhere(feature[..., 0] < 1)[:, 0]
                spon_idx = np.argwhere(feature[..., -1] > 0)[:, 0]
                wash_idx = np.unique(np.concatenate([spike_idx, spon_idx]))
                good_idx = np.delete(np.arange(dynamic.shape[0]), wash_idx).astype(np.int32)

            _end = min(num, cnt + good_idx.shape[0])
            param_arr[cnt:_end] = param[good_idx[:_end - cnt]]
            dynamic_arr[cnt:_end] = dynamic[good_idx[:_end - cnt]]
            feature_arr[cnt:_end] = feature[good_idx[:_end - cnt]]
            cnt = _end
            print("NUM: {}".format(cnt))

        param_arr -= self.param_ranges[0]
        param_arr /= self.param_ranges[1] - self.param_ranges[0]

        with h5.File(store_file, 'w') as file:
            file.create_dataset('dynamic_data', data=dynamic_arr)
            file.create_dataset('param_data', data=param_arr)
            file.create_dataset('feature_data', data=feature_arr)
