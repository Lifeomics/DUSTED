import torch


def NB_loss(y_true, mean, disp, device):
    """
    Computes the Negative Binomial (NB) loss.

    Args:
        y_true (torch.Tensor): Ground truth tensor.
        mean (torch.Tensor): Predicted mean tensor.
        disp (torch.Tensor): Predicted dispersion tensor.
        device (torch.device): Device to perform the computation on.

    Returns:
        torch.Tensor: Computed NB loss.
    """
    eps = 1e-10
    r = torch.minimum(disp, torch.tensor(1e6, device=device))
    t1 = torch.lgamma(r + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + r + eps)
    t2 = (r + y_true) * torch.log(1.0 + (mean / (r + eps))) + (y_true * (torch.log(r + eps) - torch.log(mean + eps)))
    loss = torch.mean(torch.sum(t1 + t2, dim=1))
    return loss


def ZINB_loss(y_true, mean, disp, pi, device):
    """
    Computes the Zero-Inflated Negative Binomial (ZINB) loss.

    Args:
        y_true (torch.Tensor): Ground truth tensor.
        mean (torch.Tensor): Predicted mean tensor.
        disp (torch.Tensor): Predicted dispersion tensor.
        pi (torch.Tensor): Predicted zero-inflation probability tensor.
        device (torch.device): Device to perform the computation on.

    Returns:
        torch.Tensor: Computed ZINB loss.
    """
    eps = 1e-10
    r = torch.minimum(disp, torch.tensor(1e6, device=device))
    t1 = torch.lgamma(r + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + r + eps)
    t2 = (r + y_true) * torch.log(1.0 + (mean / (r + eps))) + (y_true * (torch.log(r + eps) - torch.log(mean + eps)))

    NB = t1 + t2 - torch.log(1 - pi + eps)

    z1 = torch.pow(r / (mean + r + eps), r)
    zero_inf = -torch.log(pi + (1 - pi) * z1 + eps)

    return torch.mean(torch.where(y_true < 1e-8, zero_inf, NB))