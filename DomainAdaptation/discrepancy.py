import numpy as np
import torch
from functools import partial


def mmd_loss(x, y):
    delta = x - y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y)**2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def mk_mmd_loss(x, y, num):

    if num <= 1:
        raise ValueError(
            'Num of kernels in mk_mdd_loss should be larger than 1 but got'.
            format(num))

    sigmas = np.logspace(-8, 8, num=num, base=2)
    gaussian_kernel = partial(gaussian_kernel_matrix,
                              sigmas=torch.cuda.FloatTensor(sigmas,
                                                            device=x.device))

    loss_value = maximum_mean_discrepancy(x, y, kernel=gaussian_kernel)
    loss_value = loss_value

    return loss_value
