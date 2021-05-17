#import nemo
import pickle

import numpy as np
import pandas
import yaml
import sys
sys.path.append("siren/torchmeta")
sys.path.append("siren")
from onnx import numpy_helper
from pandas import DataFrame
from copy import deepcopy
from quantize import quantize_model, weight_svd_auto_mode
import onnx
from tqdm import tqdm
from pathlib import Path
# plotting\
import skimage
import matplotlib.pyplot as plt
import holoviews as hv
import zlib
import io
#hv.notebook_extension('bokeh')
#from holoviews import opts
from bokeh.models import HoverTool
from bokeh.plotting import show
import nemo
# torch
import torch

# @title Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)



import sys
import os

sys.path.append('siren')
import dataio, meta_modules, utils, training, loss_functions, modules
from absl import app
from absl import flags
import glob, PIL
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import json
from utils import  check_metrics, check_metrics_full
def validate(model, device, coords, img, integer=False):
    model.eval()
    with torch.no_grad():
            data, target = coords.to(device), img.to(device)
            output = model(data)
            mse = torch.mean((output - target)**2)
    return mse


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

def convert_to_nn_module_in_place(net):

    for name, module in net.named_children():
        if module.__class__.__name__ == 'BatchLinear':
            linear_module = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=True if module.bias is not None else False)
            linear_module.weight.data = module.weight.data.clone()
            linear_module.bias.data = module.bias.data.clone()
            setattr(net, name, linear_module)

        elif module.__class__.__name__ == 'MetaSequential':
            new_module = convert_to_nn_module(module)
            setattr(net, name, new_module)
        else:
            if len(list(module.named_children())):
                new_module = convert_to_nn_module(module)
                setattr(net, name, new_module)

    return net

def mse_func(a,b):
    return np.mean((np.array(a, dtype='float32') - np.array(b, dtype='float32'))**2)



bitrange = [8]#list(range(4, 18, 2))

exp_root = 'exp/archive'


imglob = glob.glob('/home/yannick/KODAK/kodim21.png')
experiment_names = [ #'64hu',
                '32hu'
                #'64hul10.00001' , '64hul10.0001' ,
                #'128hu','256hu'
                #'64hul10.0001', '128hul10.0001', '128hul10.00001', '256hul10.0001', '256hul10.00001'
                ]
#experiment_names = ['64hu']#, '64hul10.00001' , '64hul10.0001']
             #   '128hu','256hu'
                #'64hul10.00001', '128hul10.0001', '128hul10.00001', '256hul10.0001', '256hul10.00001'
           #     ]NIbr
experiment_names = ['KODAK_epochs10000_lr0.0001_hdims32_hlayer4_nerf_sine_enc_scale4.0']
experiment_names = ['siren/exp/KODAK21_epochs10000_lr0.0001_hdims100_hlayer4_gauss_sine_*enc_scale4.0/']
experiment_names = [i.split('/')[-4] for i in
                    glob.glob(exp_root + "/KODAK21_*multi*_hdims[1346]*_hlayer*_nerf_sine/kodim21/checkpoints/model_best_.pth")]
#experiment_names = [i.split('/')[-4] for i in
 #                    glob.glob(exp_root + '/KODAK21_epochs10000_lr0.0001_hdims[4]*_hlayer4_nerf_sine*enc_scale10.0/kodim21/checkpoints/model_aimet_0.8.pth')]
#experiment_names = experiment_names + [i.split('/')[-4] for i in
#                     glob.glob(exp_root + '/KODAK21_epochs10000_lr0.0001_hdims[4]*_hlayer4_nerf_sine_enc_scale10.0/kodim21/checkpoints/model_aimet_0.8.pth')]
# get experiment FLAGS

i = 0
df_list = []
for experiment_name in experiment_names:
    exp_folder = os.path.join(exp_root, experiment_name)
    TRAINING_FLAGS = yaml.safe_load(open(os.path.join(exp_folder, 'FLAGS.yml'), 'r'))

    psnr_list_nemo = []
    bpp_list_nemo = []
    ssim_list_nemo = []
    psnr_list = []
    bpp_list = []
    ssim_list = []
    for im in imglob:

        image_name = im.split('/')[-1].split('.')[0]

        img_dataset = dataio.ImageFile(im)
        img = PIL.Image.open(im)
        scale = TRAINING_FLAGS['downscaling_factor']
        image_resolution = (img.size[1] // scale, img.size[0] // scale)

        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)

        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
        #hu = int(experiment_name.split('hu')[0])
        if 'encoding_scale' in TRAINING_FLAGS:
            s = TRAINING_FLAGS['encoding_scale']

        else:
            s = 0
        if 'bn' not in TRAINING_FLAGS:
            TRAINING_FLAGS['bn'] = False
        if TRAINING_FLAGS['model_type'] == 'mlp':
            model = modules.SingleBVPNet_INR(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                             sidelength=image_resolution,
                                             out_features=img_dataset.img_channels,
                                             hidden_features=TRAINING_FLAGS['hidden_dims'],
                                             num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                             batch_norm=TRAINING_FLAGS['bn'])
        elif TRAINING_FLAGS['model_type'] == 'multi_tapered':
            model = modules.MultiScale_INR(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                           sidelength=image_resolution,
                                           out_features=img_dataset.img_channels,
                                           hidden_features=TRAINING_FLAGS['hidden_dims'],
                                           num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                           tapered=True)
        elif TRAINING_FLAGS['model_type'] == 'multi':
            model = modules.MultiScale_INR(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                           sidelength=image_resolution,
                                           out_features=img_dataset.img_channels,
                                           hidden_features=TRAINING_FLAGS['hidden_dims'],
                                           num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                           tapered=False)



        model = model.to(device)
        #state_dict = torch.load('siren/experiment_scripts/logs/' + experiment_name + image_name + '/checkpoints/model_current.pth', map_location='cpu')
        state_dict = torch.load(os.path.join(exp_folder, image_name + '/checkpoints/model_best_.pth'), map_location='cpu')

        model.load_state_dict(state_dict, strict=True)
        mse, ssim, psnr = check_metrics_full(dataloader, model, image_resolution)
       # with open(os.path.join(exp_folder, 'result_best.json'), 'w') as fp:
        #    json.dump(metrics, fp)

        model= convert_to_nn_module_in_place(model)
        model.use_meta = False

        # for name, module in model.named_modules():
        #     # prune 20% of connections in all 2D-conv layers
        #     # prune 40% of connections in all linear layers
        #     if isinstance(module, torch.nn.Linear):
        #         prune.l1_unstructured(module, name='weight', amount=0.1)
        #         prune.remove(module, 'weight')

        #
        # with torch.no_grad():
        #     bpp = []
        #     psnr = []
        #     ssim = []
        #     for Q in bitrange:
        #         dummy_input_img = torch.randn(image_resolution)
        #         dummy_input_wrapper = dataio.Implicit2DWrapper(dummy_input_img, sidelength=image_resolution)
        #         dummy_input_coords = dummy_input_wrapper.mgrid
        #
        #         model_q = nemo.transform.quantize_pact(deepcopy(model), dummy_input=dummy_input_coords.to(device))
        #         model_q.change_precision(bits=Q, scale_weights=True, scale_activations=True)
        #         model_q.qd_stage(eps_in=1./255)
        #
        #         for item in coord_dataset:
        #             in_dict, gt_dict = item
        #             break
        #        # mse = validate(model_q, device, in_dict['coords'], gt_dict['img'])
        #         mse_value, ssim_value, psnr_value = check_metrics(dataloader, model_q, image_resolution)
        #         im_out = torch.clip((model_q(in_dict['coords'].cuda()) + 1) /2, 0, 1)
        #         im_out = torch.reshape(im_out, [image_resolution[0], image_resolution[1], 3])
        #         #plt.imshow(im_out.cpu())
        #         out_img = PIL.Image.fromarray(np.uint8(im_out.cpu()*255))
        #
        #         model_q.id_stage()
        #         nemo.utils.export_onnx('model.onnx',model_q, model_q, [1, 2], round_params=True) #dummy_input_coords.shape)
        #         model_l = onnx.load_model('model.onnx')
        #         size_onnx = Path('model.onnx').stat().st_size
        #         weights = model_l.graph.initializer
        #         num_param = 0
        #         weights_np = []
        #         for l in weights:
        #             w = numpy_helper.to_array(l)
        #             if Q < 9: tpe = 'int8'
        #             elif Q < 17: tpe = 'int16'
        #             else: tpe = 'int32'
        #             w = w.astype(tpe).flatten()
        #             num_param += len(w)
        #             weights_np.append(w)
        #         weights_np = np.concatenate(weights_np)
        #         comp = zlib.compress(weights_np, level=9)
        #         #np.savez_compressed('weights.npz', weights_np)
        #         #size_np = Path('weights.npz').stat().st_size
        #         size = len(comp)
        #         s_param = num_param * Q / (image_resolution[0] * image_resolution[1])
        #         s = size * 8 / (image_resolution[0] * image_resolution[1])
        #         out_img.save(experiment_name + im.split('/')[-1].split('.')[0]+ str(Q) + 'bit' + '_bpp' + str(s) + 'psnr' + str(psnr_value), format='jpeg', quality=100, subsampling=0)
        #         bpp.append(s)
        #         ssim.append(ssim_value)
        #         psnr.append(psnr_value)
        #
        #
        #     bpp_list_nemo.append(bpp)
        #     psnr_list_nemo.append(psnr)
        #     ssim_list_nemo.append(ssim)
        #model = weight_svd_auto_mode(model, comp_ratio=0.8, retrain=False)
        model_quantized, metrics, bytes = quantize_model(model, bitwidth=8)
        bpp_val = bytes * 8 / (image_resolution[0] * image_resolution[1])
        bpp_list.append([bpp_val])
        mse, ssim, psnr = metrics
        psnr_list.append([psnr])
        ssim_list.append([ssim])





    psnr_mean = np.mean(np.array(psnr_list), axis=0)
    bpp_mean = np.mean(np.array(bpp_list), axis=0)
    ssim_mean = np.mean(np.array(ssim_list), axis=0)
    metrics = {'activation': TRAINING_FLAGS['activation'], 'encoding': TRAINING_FLAGS['encoding'],
               'hidden_dims': TRAINING_FLAGS['hidden_dims'],
               'hidden_layers': TRAINING_FLAGS['hidden_layers'],'model_type': TRAINING_FLAGS['model_type'],
               'psnr':psnr_mean, 'ssim': ssim_mean, 'encoding_scale': TRAINING_FLAGS['encoding_scale'],
               'bpp': bpp_mean,
               'l1_reg': TRAINING_FLAGS['l1_reg'], 'bn': TRAINING_FLAGS['bn'] if 'bn' in TRAINING_FLAGS else False}
    df_list.append(metrics)
    df = pandas.DataFrame.from_records(df_list)
    # col = df.hidden_layers.map({4:'b', 6:'r', 8:'g', 10:'y', 12:'c'})
    # df.plot( kind = 'scatter',c=col)
    df.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='hidden_layers', colormap='viridis')

    plot_name = str(TRAINING_FLAGS['hidden_dims']) + ' units,' + str(TRAINING_FLAGS['hidden_layers']) + ' layers, ' + str(TRAINING_FLAGS['spec_reg'])# + ' l1_reg_' + 'aimet' + str(aimet)
    # plt.figure(1)
    # plt.plot(bitrange, psnr_mean, 'x', label=plot_name)

    plt.legend()
    # plt.figure(2)
    # plt.hist(weights_np, 101, label=plot_name, alpha=0.5)
    # plt.legend()

    plt.figure(3)
    plt.plot(bpp_mean, psnr_mean, 'x', label=plot_name)
    # plt.figure(3)
    # plt.plot(bpp_mean, psnr_list_nemo[0], 'o', label=plot_name + '_nemo')
    # plt.figure(4)
    # plt.plot(bpp_mean, ssim_mean, label= plot_name)

# Compute JPEG baseline
psnr_list_jpeg = []
bpp_list_jpeg = []
ssim_list_jpeg = []
for im in imglob:
    img = PIL.Image.open(im)
    img = img.resize((img.size[0]//2, img.size[1]//2))

    bpp = []
    psnr = []
    lpips = []
    ssim = []

    for quality in list(range(1, 10, 2)) + list(range(10, 110, 10)):
        out = io.BytesIO()
        img.save(out, format='jpeg', quality=quality, subsampling=0)

        bpp_val = out.tell() * 8 / (img.size[0] * img.size[1])
        bpp.append(bpp_val)

        img_tilde = PIL.Image.open(out)
        psnr_val = 10 * np.log10(256 ** 2 / mse_func(img, img_tilde))
        ssim_val = skimage.measure.compare_ssim(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'),
                                         multichannel=True, data_range=256)
        psnr.append(psnr_val)
        ssim.append(ssim_val)
        img.save("jpeg_ref/" + im.split('/')[-1].split('.')[0] + "_Q" + str(quality) + '_bpp' + str(bpp_val) + 'psnr' + str(psnr_val) + ".jpg", format='jpeg',
                 quality=quality, subsampling=0)
        # lpips_out = lpips_tf.lpips(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'))
        # lpips.append(sess.run(lpips_out))
        # print(lpips)
    bpp_list_jpeg.append(bpp)
    psnr_list_jpeg.append(psnr)
    ssim_list_jpeg.append(ssim)
    # lpips_list.append(lpips)

bpp_mean_jpeg = np.mean(np.array(bpp_list_jpeg), axis=0)
psnr_mean_jpeg = np.mean(np.array(psnr_list_jpeg), axis=0)
ssim_mean_jpeg = np.mean(np.array(ssim_list_jpeg), axis=0)

plt.figure(3)
plt.plot(bpp_mean_jpeg, psnr_mean_jpeg, label='JPEG')
plt.xlabel('bpp')
plt.ylabel('PSNR')
plt.xlim([0, 2])
plt.ylim([10, 35])
plt.legend()


# plt.figure(4)
# plt.plot(bpp_mean_jpeg, ssim_mean_jpeg, label='JPEG')
# plt.xlabel('bpp')
# plt.xlim([0, 5])
# plt.ylabel('SSIM')
# plt.legend()
out_dict = df.to_dict()
run_name='multi'
with open("plots/" + run_name + ".pickle", "wb") as output_file:
    pickle.dump(out_dict, output_file)


plt.show()
print('JPEG: ')
print(psnr_list_jpeg)
print('INR: ')
print(psnr_list)


