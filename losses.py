import torch
import torch.nn.functional as F


def image_log_mse(model_output, gt):
    return {'img_loss': torch.log10(((model_output['model_out'] - gt['img']) ** 2).mean())}

def model_l1(model, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return {'l1_loss': l1_lambda * l1_norm}

def model_l1_diff(ref_model, model, l1_lambda):
    l1_norm = sum((p - ref_p).abs().sum() for (p, ref_p) in zip(model.parameters(), ref_model.parameters()))
    return {'l1_loss': l1_lambda * l1_norm}



def spectral_norm_loss(model, spec_lambda):
    weight_matrices = filter(lambda n: '.weight' in n[0], model.named_parameters())
    sigma_sum = 0
    for name, mat in weight_matrices:
        sigma_sum += _spectral_norm(mat)
    return {'spec_loss': sigma_sum * spec_lambda}


def _L2Normalize(v, eps=1e-12):
    return v/(torch.norm(v) + eps)

def _spectral_norm(W, u=None, Num_iter=10):
    '''
    Spectral Norm of a Matrix is its maximum singular value.
    This function employs the Power iteration procedure to
    compute the maximum singular value.
    (https://antixk.github.io/blog/lipschitz-wgan/)
    ---------------------
    :param W: Input(weight) matrix - autograd.variable
    :param u: Some initial random vector - FloatTensor
    :param Num_iter: Number of Power Iterations
    :return: Spectral Norm of W, orthogonal vector _u
    '''
    if not Num_iter >= 1:
        raise ValueError("Power iteration must be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0,1).cuda()
    # Power iteration
    for _ in range(Num_iter):
        v = _L2Normalize(torch.matmul(u, W.data))
        u = _L2Normalize(torch.matmul(v, torch.transpose(W.data,0, 1)))
    sigma = torch.sum(F.linear(u, torch.transpose(W.data, 0,1)) * v)
    return sigma