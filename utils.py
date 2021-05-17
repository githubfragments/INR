import torch
from torch.utils.data import DataLoader
from siren import dataio
import numpy as np
import skimage

def convert_to_nn_module(net):
    out_net = torch.nn.Sequential()
    for name, module in net.named_children():
        if module.__class__.__name__ == 'BatchLinear':
            linear_module = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=True if module.bias is not None else False)
            linear_module.weight.data = module.weight.data.clone()
            linear_module.bias.data = module.bias.data.clone()
            out_net.add_module(name, linear_module)
        elif module.__class__.__name__ == 'Sine':
            out_net.add_module(name, module)

        elif module.__class__.__name__ == 'MetaSequential':
            new_module = convert_to_nn_module(module)
            out_net.add_module(name, new_module)
        else:
            if len(list(module.named_children())):
                out_net.add_module(name, convert_to_nn_module(module))
            else: out_net.add_module(name, module)
    return out_net

def check_metrics_full(test_loader: DataLoader, model: torch.nn.Module, image_resolution):
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            predictions = model(model_input)
            gt_img = dataio.lin2img(gt['img'], image_resolution)
            pred_img = dataio.lin2img(predictions['model_out'], image_resolution)
            pred_img = pred_img.detach().cpu().numpy()[0]
            gt_img = gt_img.detach().cpu().numpy()[0]
            p = pred_img.transpose(1, 2, 0)
            trgt = gt_img.transpose(1, 2, 0)
            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5
            mse  = skimage.measure.compare_mse(p, trgt)
            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

def check_metrics(test_loader: DataLoader, model: torch.nn.Module, image_resolution):
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            predictions = model(model_input['coords'])
            gt_img = dataio.lin2img(gt['img'], image_resolution)
            pred_img = dataio.lin2img(predictions, image_resolution)
            pred_img = pred_img.detach().cpu().numpy()[0]
            gt_img = gt_img.detach().cpu().numpy()[0]
            p = pred_img.transpose(1, 2, 0)
            trgt = gt_img.transpose(1, 2, 0)
            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5
            mse  = skimage.measure.compare_mse(p, trgt)
            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)


    return mse, ssim, psnr

