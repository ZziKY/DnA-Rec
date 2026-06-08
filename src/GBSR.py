import torch


def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0, 1)) - 1) / sigma)


def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = (torch.trace(Kxy) / m ** 2
         + torch.mean(Kx) * torch.mean(Ky)
         - 2 * torch.mean(Kxy) / m)
    return h * (m / (m - 1)) ** 2
