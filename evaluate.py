import statistics

import nemo
import numpy as np
import pandas
import yaml
from onnx import numpy_helper
from pandas import DataFrame
from copy import deepcopy

import onnx
from tqdm import tqdm
from pathlib import Path
# plotting\
import skimage
import matplotlib.pyplot as plt
import holoviews as hv
import zlib
import io
from holoviews import opts
from bokeh.models import HoverTool
from bokeh.plotting import show

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
from utils import check_metrics, check_metrics_full

def validate(model, device, coords, img, integer=False):
    model.eval()
    with torch.no_grad():
            data, target = coords.to(device), img.to(device)
            output = model(data)
            mse = torch.mean((output - target)**2)
    return mse


exp_root = 'exp/maml vs nomaml'
psnr_list = []
bpp_list = []
bitrange = list(range(2, 18, 2))




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
           #     ]
experiment_names = ['KODAK21_epochs10000_lr0.0001_hdims100_hlayer4_gauss_sine_enc_scale4.0']
experiment_names =[i.split('/')[-4] for i in glob.glob(exp_root + '/KODAK21_epochs10000*/kodim21/checkpoints/model_best_.pth')]
# get experiment FLAGS

i = 0
df_list = []
for experiment_name in experiment_names:
    mses = []
    psnrs = []
    ssims = []
    exp_folder = os.path.join(exp_root, experiment_name)
    TRAINING_FLAGS = yaml.safe_load(open(os.path.join(exp_folder, 'FLAGS.yml'), 'r'))

    if TRAINING_FLAGS['model_type'] not in ['mixture', 'mlp']: continue
    # if TRAINING_FLAGS['hidden_dims'] not in [32, 48, 64, 128]: continue
    if TRAINING_FLAGS['hidden_layers'] > 4: continue
    #if TRAINING_FLAGS['model_type'] == 'mlp' and 'ff_dims' in TRAINING_FLAGS and int(TRAINING_FLAGS['ff_dims'][0]) not in [4,6,8]: continue
    #
    # if TRAINING_FLAGS['model_type'] == 'multi_tapered':
    #     TRAINING_FLAGS['model_type'] = 'multi'
    # elif TRAINING_FLAGS['model_type'] == 'multi':
    #     TRAINING_FLAGS['model_type'] = 'multi_tapered'
    # yaml.dump(TRAINING_FLAGS, open(os.path.join(exp_folder, 'FLAGS.yml'), 'w'))
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
        if 'intermediate_losses' not in TRAINING_FLAGS:
            TRAINING_FLAGS['intermediate_losses'] = False
            if 'phased' not in TRAINING_FLAGS:
                TRAINING_FLAGS['phased'] = False
        if 'ff_dims' not in TRAINING_FLAGS:
            TRAINING_FLAGS['ff_dims'] = None
        if 'num_components' not in TRAINING_FLAGS:
            TRAINING_FLAGS['num_components'] = 1
        is_maml = False
        if 'maml_epochs' in TRAINING_FLAGS:
            is_maml = True
        if TRAINING_FLAGS['model_type'] == 'mlp':
            model = modules.SingleBVPNet_INR(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                             sidelength=image_resolution,
                                             out_features=img_dataset.img_channels,
                                             hidden_features=TRAINING_FLAGS['hidden_dims'],
                                             num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                             batch_norm=TRAINING_FLAGS['bn'], ff_dims=TRAINING_FLAGS['ff_dims'])
        elif TRAINING_FLAGS['model_type'] == 'multi_tapered':
            model = modules.MultiScale_INR(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                             sidelength=image_resolution,
                                             out_features=img_dataset.img_channels,
                                             hidden_features=TRAINING_FLAGS['hidden_dims'],
                                             num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                           tapered=True, ff_dims=TRAINING_FLAGS['ff_dims'])
        elif TRAINING_FLAGS['model_type'] == 'multi':
            model = modules.MultiScale_INR(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                             sidelength=image_resolution,
                                             out_features=img_dataset.img_channels,
                                             hidden_features=TRAINING_FLAGS['hidden_dims'],
                                             num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                            tapered=False, ff_dims=TRAINING_FLAGS['ff_dims'])
        elif TRAINING_FLAGS['model_type'] == 'parallel':
            model = modules.Parallel_INR(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                             sidelength=image_resolution,
                                             out_features=img_dataset.img_channels,
                                             hidden_features=[TRAINING_FLAGS['hidden_dims'] // 4, TRAINING_FLAGS['hidden_dims'] // 2,
                                                          TRAINING_FLAGS['hidden_dims']],
                                             num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s)
        elif TRAINING_FLAGS['model_type'] == 'mixture':
            model = modules.INR_Mixture(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                             sidelength=image_resolution,
                                             out_features=img_dataset.img_channels,
                                             hidden_features=TRAINING_FLAGS['hidden_dims'],
                                             num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                         batch_norm=TRAINING_FLAGS['bn'], ff_dims=TRAINING_FLAGS['ff_dims'], num_components=TRAINING_FLAGS['num_components'])

        model = model.to(device)
        #state_dict = torch.load('siren/experiment_scripts/logs/' + experiment_name + image_name + '/checkpoints/model_current.pth', map_location='cpu')
        state_dict = torch.load(os.path.join(exp_folder, image_name + '/checkpoints/model_best_.pth'), map_location='cpu')
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            print('Failed to load model ' + experiment_name + image_name)
            continue
        # model = torch.load(os.path.join(exp_folder, image_name + '/checkpoints/model_aimet_.pth'))
        # model = model.cuda()
        #mse, ssim, psnr = check_metrics(dataloader, model, image_resolution)
        mse, ssim, psnr = check_metrics_full(dataloader, model, image_resolution)
        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(ssim)
        num_params = sum(p.numel() for p in model.parameters() )#if p.requires_grad)


    if len(mses) > 0:
        metrics = {'activation': TRAINING_FLAGS['activation'] , 'model_type': TRAINING_FLAGS['model_type'], 'encoding': TRAINING_FLAGS['encoding'], 'hidden_dims': TRAINING_FLAGS['hidden_dims'],
                   'hidden_layers': TRAINING_FLAGS['hidden_layers'], 'mse': statistics.mean(mses),
                   'psnr': statistics.mean(psnrs), 'ssim': statistics.mean(ssims), 'encoding_scale': s, 'num_params': num_params,
                   'l1_reg': TRAINING_FLAGS['l1_reg'], 'bn': TRAINING_FLAGS['bn'], 'phased' : TRAINING_FLAGS['phased'], 'intermediate_losses' : TRAINING_FLAGS['intermediate_losses'], 'ff_dims': TRAINING_FLAGS['ff_dims'],
                   'num_components': TRAINING_FLAGS['num_components'], 'is_maml': is_maml}

        # with open(os.path.join(exp_folder, 'result_best.json'), 'w') as fp:
        #         json.dump(metrics, fp)
        df_list.append(metrics)
df = pandas.DataFrame.from_records(df_list)
#col = df.hidden_layers.map({4:'b', 6:'r', 8:'g', 10:'y', 12:'c'})
#col = df.model_type.map({'multi':'b', 'mlp':'r', 'parallel':'g', 'multi_tapered':'y', 'mixture':'k'})
labels = df.model_type.map({'multi':'multi', 'mlp':'mlp', 'parallel':'parallel', 'multi_tapered':'multi_tapered'})
#df.plot( kind = 'scatter',c=col)
df_maml = df[df['is_maml'] == True]
df_mlp = df[df['is_maml'] == False]
ax = df_mlp.plot.scatter(x='num_params', y='psnr', xlabel='Parameters', ylabel='PSNR',
                     c='r', label='random')
df_maml.plot.scatter(x='num_params', y='psnr', xlabel='Parameters', ylabel='PSNR',
                     c='b', label = 'maml', ax=ax)
plt.legend()
plt.show()
