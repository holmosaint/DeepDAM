import numpy as np
import torch
from functools import partial


def mmd_loss(x, y):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss between two sets of samples.

    The MMD loss is a measure of the difference between the distributions of two sets of samples.
    It is commonly used in domain adaptation and generative models to ensure that the generated
    samples have a similar distribution to the real samples.

    Args:
        x (torch.Tensor): A tensor of shape (n_samples, n_features) representing the first set of samples.
        y (torch.Tensor): A tensor of shape (n_samples, n_features) representing the second set of samples.

    Returns:
        torch.Tensor: A scalar tensor representing the MMD loss.
    """
    delta = x - y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def pairwise_distance(x, y):
    """
    Computes the pairwise squared Euclidean distance between two matrices.

    Parameters:
    x (torch.Tensor): A 2D tensor of shape (n_samples_x, n_features).
    y (torch.Tensor): A 2D tensor of shape (n_samples_y, n_features).

    Returns:
    torch.Tensor: A 2D tensor of shape (n_samples_x, n_samples_y) containing the pairwise squared Euclidean distances.

    Raises:
    ValueError: If either input is not a 2D tensor.
    ValueError: If the number of features (columns) in x and y do not match.
    """

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
    """
    Computes the Gaussian kernel matrix between two sets of vectors.

    Args:
        x (torch.Tensor): A tensor of shape (n_samples_x, n_features) representing the first set of vectors.
        y (torch.Tensor): A tensor of shape (n_samples_y, n_features) representing the second set of vectors.
        sigmas (torch.Tensor): A tensor of shape (n_sigmas,) representing the bandwidths for the Gaussian kernels.

    Returns:
        torch.Tensor: A tensor of shape (n_samples_x, n_samples_y) representing the Gaussian kernel matrix.
    """

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples.

    The MMD is a distance measure between the distributions of two sets of samples.
    It is computed using a kernel function, typically a Gaussian kernel.

    Args:
        x (torch.Tensor): A tensor of samples from the first distribution.
        y (torch.Tensor): A tensor of samples from the second distribution.
        kernel (callable, optional): A function that computes the kernel matrix. 
                                     Defaults to `gaussian_kernel_matrix`.

    Returns:
        torch.Tensor: A scalar tensor representing the MMD between the two sets of samples.
    """

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def mk_mmd_loss(x, y, num):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss between two sets of samples.

    This function calculates the MMD loss using a mixture of Gaussian kernels.
    The number of kernels is specified by the `num` parameter, which must be greater than 1.

    Parameters:
    x (torch.Tensor): The first set of samples.
    y (torch.Tensor): The second set of samples.
    num (int): The number of Gaussian kernels to use. Must be greater than 1.

    Returns:
    torch.Tensor: The computed MMD loss value.

    Raises:
    ValueError: If `num` is less than or equal to 1.
    """

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
