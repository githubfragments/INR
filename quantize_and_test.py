import copy
import pickle
import zlib

import numpy as np
import pandas
import yaml
import sys

from losses import model_l1

sys.path.append("siren/torchmeta")
sys.path.append("siren")
import skimage
import matplotlib.pyplot as plt
import io
import torch

# @title Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)
import sys
import torch
import os
from tqdm import tqdm
from aimet_torch.qc_quantize_op import QcPostTrainingWrapper
from aimet_torch.quantsim import QuantizationSimModel
import dataio, meta_modules, utils, training, loss_functions, modules
from modules import Sine, ImageDownsampling, PosEncodingNeRF, FourierFeatureEncodingPositional, \
    FourierFeatureEncodingGaussian
from absl import app
from absl import flags
import glob, PIL
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import json
from utils import check_metrics, check_metrics_full
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from quantize_utils import convert_to_nn_module, convert_to_nn_module_in_place
from aimet_torch.save_utils import SaveUtils
from aimet_common.defs import QuantScheme
from aimet_torch.meta import connectedgraph_utils


def mse_func(a, b):
    return np.mean((np.array(a, dtype='float32') - np.array(b, dtype='float32')) ** 2)


flags.DEFINE_string('data_root',
                    '/home/yannick',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')
flags.DEFINE_string('exp_glob',
                    'KODAK21*',
                    'regular expression to match experiment name')
flags.DEFINE_enum('dataset', 'KODAK21',
                  ['KODAK', 'KODAK21'],
                  'Dataset used during retraining.')
flags.DEFINE_enum('difference_encoding', 'same',
                  ['same', 'adjusted', 'off'],
                  'Difference encoding mode')
flags.DEFINE_integer('epochs',
                     10000,
                     'Maximum number of epochs during retraining.',
                     lower_bound=1)
flags.DEFINE_float('lr',
                   1e-06,
                   'Learning rate used during retraining.',
                   lower_bound=0.0)
flags.DEFINE_float('l1_reg',
                   0.0,
                   'L1 weight regularization used during retraining.',
                   lower_bound=0.0)

flags.DEFINE_integer('bitwidth',
                     8,
                     'bitwidth used for Quantization',
                     lower_bound=1)
flags.DEFINE_bool('adaround',
                  True,
                  'use adative rounding post quanitzatition')
flags.DEFINE_bool('retrain',
                  True,
                  'use retraining post quanitzatition')
flags.DEFINE_float('adaround_reg', 0.001, 'regularizing parameter for adaround')
flags.DEFINE_integer('adaround_iterations', 500, 'Number of adaround iterations')

FLAGS = flags.FLAGS


class AimetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (self.dataset[idx][0]['coords'].unsqueeze(0), self.dataset[idx][1]['img'])


def apply_adaround(model, sim, dataloader, exp_folder, image_name, input_shape, bitwidth, layerwise_bitwidth=None,
                   adaround_reg=0.01, adaround_iterations=500):
    dummy_in = ((torch.rand(input_shape).unsqueeze(0) * 2) - 1).cuda()
    params = AdaroundParameters(data_loader=dataloader, num_batches=1, default_num_iterations=adaround_iterations,
                                default_reg_param=adaround_reg, default_beta_range=(20, 2))

    # Compute only param encodings
    Adaround._compute_param_encodings(sim)

    # Get the module - activation function pair using ConnectedGraph
    module_act_func_pair = connectedgraph_utils.get_module_act_func_pair(model, dummy_in)

    Adaround._adaround_model(model, sim, module_act_func_pair, params, dummy_in)

    # Update every module (AdaroundSupportedModules) weight with Adarounded weight (Soft rounding)
    Adaround._update_modules_with_adarounded_weights(sim)
    path = os.path.join(exp_folder, image_name)

    filename_prefix = 'adaround'
    # Export quantization encodings to JSON-formatted file
    Adaround._export_encodings_to_json(path, filename_prefix, sim)
    SaveUtils.remove_quantization_wrappers(sim.model)
    adarounded_model = sim.model

    sim = get_quant_sim(model=adarounded_model, input_shape=input_shape, bitwidth=bitwidth,
                        layerwise_bitwidth=layerwise_bitwidth)
    sim.set_and_freeze_param_encodings(encoding_path=os.path.join(path, filename_prefix + '.encodings'))
    return sim


def get_quant_sim(model, input_shape, bitwidth, layerwise_bitwidth=None):
    dummy_in = ((torch.rand(input_shape).unsqueeze(0) * 2) - 1).cuda()
    sim = QuantizationSimModel(model, default_param_bw=bitwidth,
                               default_output_bw=31, dummy_input=dummy_in)
    modules_to_exclude = (
        Sine, ImageDownsampling, PosEncodingNeRF, FourierFeatureEncodingPositional, FourierFeatureEncodingGaussian)
    excl_layers = []
    for mod in sim.model.modules():
        if isinstance(mod, QcPostTrainingWrapper) and isinstance(mod._module_to_wrap, modules_to_exclude):
            excl_layers.append(mod)

    sim.exclude_layers_from_quantization(excl_layers)
    i = 0
    for name, mod in sim.model.named_modules():
        if isinstance(mod, QcPostTrainingWrapper):
            mod.output_quantizer.enabled = False
            mod.input_quantizer.enabled = False
            weight_quantizer = mod.param_quantizers['weight']
            bias_quantizer = mod.param_quantizers['bias']

            weight_quantizer.use_symmetric_encodings = True
            bias_quantizer.use_symmetric_encodings = True
            if torch.count_nonzero(mod._module_to_wrap.bias.data):
                mod.param_quantizers['bias'].enabled = True
            if layerwise_bitwidth:
                mod.param_quantizers['bias'].bitwidth = layerwise_bitwidth[i]
                mod.param_quantizers['weight'].bitwidth = layerwise_bitwidth[i]
                i += 1
    return sim


def apply_quantization(sim):
    quantized_dict = {}
    state_dict = {}
    for name, module in sim.model.named_modules():
        if isinstance(module, QcPostTrainingWrapper) and isinstance(module._module_to_wrap, torch.nn.Linear):
            weight_quantizer = module.param_quantizers['weight']
            bias_quantizer = module.param_quantizers['bias']
            weight_quantizer.enabled = True
            bias_quantizer.enabled = True
            wrapped_linear = module._module_to_wrap
            weight = wrapped_linear.weight
            bias = wrapped_linear.bias

            state_dict[name + '.weight'] = weight_quantizer.quantize_dequantize(weight,
                                                                                weight_quantizer.round_mode).cpu().detach()
            assert (len(torch.unique(state_dict[name + '.weight'])) <= 2 ** weight_quantizer.bitwidth)
            state_dict[name + '.bias'] = bias_quantizer.quantize_dequantize(bias,
                                                                            bias_quantizer.round_mode).cpu().detach()
            assert (len(torch.unique(state_dict[name + '.bias'])) <= 2 ** bias_quantizer.bitwidth)
            quantized_weight = weight_quantizer.quantize(weight,
                                                         weight_quantizer.round_mode).cpu().detach().numpy() + weight_quantizer.encoding.offset
            quantized_bias = bias_quantizer.quantize(bias,
                                                     bias_quantizer.round_mode).cpu().detach().numpy() + bias_quantizer.encoding.offset
            quantized_dict[name] = {'weight': {'data': quantized_weight, 'encoding': weight_quantizer.encoding},
                                    'bias': {'data': quantized_bias, 'encoding': bias_quantizer.encoding}}

    weights_np = []
    for l in quantized_dict.values():
        w = l['weight']['data']
        b = l['bias']['data']
        Q = l['weight']['encoding'].bw
        if Q < 9:
            tpe = 'int8'
        elif Q < 17:
            tpe = 'int16'
        else:
            tpe = 'int32'
        w = w.astype(tpe).flatten()
        weights_np.append(w)

        if l['bias']['encoding']:
            Q = l['bias']['encoding'].bw
            if Q < 9:
                tpe = 'int8'
            elif Q < 17:
                tpe = 'int16'
            else:
                tpe = 'int32'
            b = b.astype(tpe).flatten()
            weights_np.append(b)
    weights_np = np.concatenate(weights_np)
    comp = zlib.compress(weights_np, level=9)
    return comp, state_dict





def retrain_model(model, train_dataloader, epochs, loss_fn, lr, l1_reg, image_resolution,
                  randomize_quant_wrappers=False,
                  weight_loss_weight=0):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    best_mse = 1
    use_amp = False
    q_wrapper_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, QcPostTrainingWrapper):
            q_wrapper_list.append(mod)
    N = len(q_wrapper_list)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        for epoch in range(epochs):
            if randomize_quant_wrappers:
                r = torch.rand(N)
                for i, q in enumerate(q_wrapper_list):
                    if r[i] > 0.5:
                        q.param_quantizers['weight'].enabled = True
                        q.param_quantizers['bias'].enabled = True
                    else:
                        q.param_quantizers['weight'].enabled = False
                        q.param_quantizers['bias'].enabled = False

            for step, (model_input, gt) in enumerate(train_dataloader):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}
                    model_output = {}
                    model_output['model_out'] = model(model_input['coords'])
                    losses = loss_fn(model_output, gt)
                    l1_loss = model_l1(model, l1_reg)
                    losses = {**losses, **l1_loss}

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss
                    weight_loss = 0

                    if weight_loss_weight:
                        for i, q in enumerate(q_wrapper_list):
                            weight_quantizer = q.param_quantizers['weight']
                            bias_quantizer = q.param_quantizers['bias']

                            wrapped_linear = q._module_to_wrap
                            weight = copy.deepcopy(wrapped_linear.weight)
                            bias = copy.deepcopy(wrapped_linear.bias)

                            weight_dequant = weight_quantizer.quantize_dequantize(weight, weight_quantizer.round_mode)
                            bias_dequant = bias_quantizer.quantize_dequantize(bias, bias_quantizer.round_mode)

                            weight_diff = torch.abs(weight_dequant - wrapped_linear.weight)
                            bias_diff = torch.abs(bias_dequant - wrapped_linear.bias)

                            weight_loss += torch.sum(weight_diff) + torch.sum(bias_diff)

                    train_loss = train_loss + weight_loss_weight * weight_loss
                    optim.zero_grad()
                    scaler.scale(train_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    pbar.update(1)

            # make sure all quanitzation wrappers are active for performance evaluation
            for i, q in enumerate(q_wrapper_list):
                q.param_quantizers['weight'].enabled = True
                q.param_quantizers['bias'].enabled = True

            m = check_metrics(train_dataloader, model, image_resolution)
            mse, ssim, psnr = m
            if mse < best_mse:
                best_state_dict = copy.deepcopy(model.state_dict())
                best_mse = mse

    model.load_state_dict(best_state_dict, strict=True)
    return model


def quantize_model(model, coord_dataset, bitwidth=8, layerwise_bitwidth=None, retrain=True, epochs=300, ref_model=None,
                   flags=None,
                   adaround=False, lr=0.00000001, adaround_reg=0.01, adaround_iterations=500, exp_folder=None,
                   image_name=None, difference_encoding='same'):
    input_shape = coord_dataset.mgrid.shape
    image_resolution = coord_dataset.sidelength
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True,
                            num_workers=0)
    aimet_dataloader = DataLoader(AimetDataset(coord_dataset), shuffle=True, batch_size=1, pin_memory=True,
                                  num_workers=0)

    sim = get_quant_sim(model=model, input_shape=input_shape, bitwidth=bitwidth, layerwise_bitwidth=layerwise_bitwidth)
    res = check_metrics(dataloader, sim.model, image_resolution)
    print('After Quantization: ', res)
    if adaround:
        sim = apply_adaround(model=model, sim=sim, dataloader=aimet_dataloader, exp_folder=exp_folder,
                             image_name=image_name,
                             input_shape=input_shape, bitwidth=bitwidth, layerwise_bitwidth=layerwise_bitwidth,
                             adaround_reg=adaround_reg, adaround_iterations=adaround_iterations)
        res = check_metrics(dataloader, sim.model, image_resolution)
        print('After Adaround: ', res)
    if retrain:
        loss_fn = partial(loss_functions.image_mse, None)
        retrained_model = retrain_model(model=sim.model, train_dataloader=dataloader, epochs=epochs, loss_fn=loss_fn,
                                        lr=lr,
                                        l1_reg=flags['l1_reg'] if flags is not None else 0,
                                        image_resolution=image_resolution)

        res = check_metrics(dataloader, retrained_model, image_resolution)
        model = retrained_model
        print('After retraining: ', res)

    def evaluate_model(model: torch.nn.Module, eval_iterations: int, use_cuda: bool = False) -> float:
        """
        :param model: Model to evaluate
        :param eval_iterations: Number of iterations to use for evaluation.
                None for entire epoch.
        :param use_cuda: If true, evaluate using gpu acceleration
        :return: single float number (accuracy) representing model's performance
        """
        mse, ssim, psnr = check_metrics(dataloader, model, image_resolution)

        return psnr

    # Compute the difference for each parameter
    if ref_model is not None and difference_encoding=='same':
        ref_sim  = get_quant_sim(model=convert_to_nn_module(ref_model), input_shape=input_shape, bitwidth=bitwidth, layerwise_bitwidth=layerwise_bitwidth)
        new_state_dict = copy.deepcopy(sim.model.state_dict())

        sim.model.load_state_dict(ref_sim.model.state_dict())
        _, ref_state_dict_quantized = apply_quantization(sim)
        lis = [[i, j, a, b] for i, a in ref_state_dict_quantized.items() for j, b in new_state_dict.items() if
               i == j.replace('._module_to_wrap', '')]
        # lis = [[i, j, a, b] for i, a in ref_model.named_parameters() for j, b in sim.model.named_parameters() if
        #        i == j.replace('._module_to_wrap', '')]
        for module in lis:
            new_state_dict[module[1]] = module[3] - module[2].cuda()

        sim.model.load_state_dict(new_state_dict)
        #sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=1)

        #ref_model_state_dict = ref_model.state_dict()

        comp, update_state_dict = apply_quantization(sim)
        # for key, value in ref_model_state_dict.items():
        #     ref_model_state_dict[key] = value + update_state_dict[key].cuda()
        # state_dict = ref_model_state_dict
        final_state_dict = {}
        for key, value in ref_state_dict_quantized.items():
            final_state_dict[key] = value + update_state_dict[key]
        state_dict = final_state_dict

    elif ref_model is not None and difference_encoding=='adjusted':

        new_state_dict = copy.deepcopy(sim.model.state_dict())
        lis = [[i, j, a, b] for i, a in ref_model.named_parameters() for j, b in sim.model.named_parameters() if
               i == j.replace('._module_to_wrap', '')]
        for module in lis:
            new_state_dict[module[1]] = module[3] - module[2].cuda()

        sim.model.load_state_dict(new_state_dict)
        sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=1)
        ref_model_state_dict = ref_model.state_dict()

        comp, update_state_dict = apply_quantization(sim)
        for key, value in ref_model_state_dict.items():
            ref_model_state_dict[key] = value + update_state_dict[key].cuda()
        state_dict = ref_model_state_dict

    else:
        comp, state_dict = apply_quantization(sim)
    print(len(comp))
    return model, res, len(comp), state_dict

def get_quant_config_name():


    # create exp folder
    name = 'bw' + str(FLAGS.bitwidth)

    if FLAGS.adaround:
        name = '_'.join([name, 'adaround_iter', str(FLAGS.adaround_iterations),'adaround_reg', str(FLAGS.adaround_reg)])
    if FLAGS.retrain:
        name = '_'.join([name, 'retrain_epochs' + str(FLAGS.epochs), 'retrain_lr' + str(FLAGS.lr)])
    if FLAGS.difference_encoding == 'adjusted':
        name = '_'.join([name, 'enc_adjusted'])


    return name


def main(_):
    imglob = glob.glob(os.path.join(FLAGS.data_root, FLAGS.dataset, '*'))

    df_list = []
    exp_glob = glob.glob(os.path.join(FLAGS.exp_root, FLAGS.exp_glob))
    for exp_folder in exp_glob:
        TRAINING_FLAGS = yaml.safe_load(open(os.path.join(exp_folder, 'FLAGS.yml'), 'r'))

        for im in imglob:
            image_name = im.split('/')[-1].split('.')[0]
            img_dataset = dataio.ImageFile(im)
            img = PIL.Image.open(im)
            scale = TRAINING_FLAGS['downscaling_factor']
            image_resolution = (img.size[1] // scale, img.size[0] // scale)
            coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)

            dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True,
                                    num_workers=0)

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
            elif TRAINING_FLAGS['model_type'] == 'mixture':
                model = modules.INR_Mixture(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                            sidelength=image_resolution,
                                            out_features=img_dataset.img_channels,
                                            hidden_features=TRAINING_FLAGS['hidden_dims'],
                                            num_hidden_layers=TRAINING_FLAGS['hidden_layers'], encoding_scale=s,
                                            batch_norm=TRAINING_FLAGS['bn'], ff_dims=TRAINING_FLAGS['ff_dims'],
                                            num_components=TRAINING_FLAGS['num_components'])

            model = model.to(device)
            try:
                state_dict = torch.load(os.path.join(exp_folder, image_name + '/checkpoints/model_best_.pth'),
                                        map_location='cpu')
            except:
                continue

            model.load_state_dict(state_dict, strict=True)
            mse, ssim, psnr = check_metrics_full(dataloader, model, image_resolution)

            try:
                ref_state_dict = torch.load(os.path.join(exp_folder, 'model_maml.pth'),
                                            map_location='cpu')
                ref_model = copy.deepcopy(model)
                ref_model.load_state_dict(ref_state_dict, strict=True)
            except:
                ref_model = None
            if TRAINING_FLAGS['model_type'] == 'mlp':
                model = convert_to_nn_module(model)
            else:
                model = convert_to_nn_module_in_place(model)
                model.use_meta = False

            model_quantized, metrics, bytes, state_dict = quantize_model(model=model, coord_dataset=coord_dataset,
                                                                         bitwidth=FLAGS.bitwidth,
                                                                         retrain=FLAGS.retrain, epochs=FLAGS.epochs,
                                                                         lr=FLAGS.lr, ref_model=ref_model,
                                                                         adaround=FLAGS.adaround,
                                                                         adaround_iterations=FLAGS.adaround_iterations,
                                                                         adaround_reg=FLAGS.adaround_reg,
                                                                         exp_folder=exp_folder,
                                                                         image_name=image_name,
                                                                         difference_encoding=FLAGS.difference_encoding)
            model.load_state_dict(state_dict, strict=True)
            metrics = check_metrics(dataloader, model, image_resolution)
            print('Final metrics: ', metrics)
            bpp_val = bytes * 8 / (image_resolution[0] * image_resolution[1])
            mse, ssim, psnr = metrics
            metrics_dict = {'activation': TRAINING_FLAGS['activation'], 'training_epochs': TRAINING_FLAGS['epochs'],
                            'encoding': TRAINING_FLAGS['encoding'], 'training_lr': TRAINING_FLAGS['lr'],
                            'hidden_dims': TRAINING_FLAGS['hidden_dims'],
                            'hidden_layers': TRAINING_FLAGS['hidden_layers'],
                            'model_type': TRAINING_FLAGS['model_type'],
                            'psnr': psnr.item(), 'ssim': ssim.item(), 'mse': mse.item(), 'encoding_scale': TRAINING_FLAGS['encoding_scale'],
                            'bpp': bpp_val,
                            'l1_reg': TRAINING_FLAGS['l1_reg'],
                            'bn': TRAINING_FLAGS['bn'] if 'bn' in TRAINING_FLAGS else False,
                            'phased': TRAINING_FLAGS['phased'],
                            'intermediate_losses': TRAINING_FLAGS['intermediate_losses'],
                            'ff_dims': TRAINING_FLAGS['ff_dims'], 'num_components': TRAINING_FLAGS['num_components'],
                            'retrain_epochs': FLAGS.epochs, 'retrain': FLAGS.retrain,
                            'adaround': FLAGS.adaround, 'adaround_reg': FLAGS.adaround_reg,
                            'adaround_iterations': FLAGS.adaround_iterations, 'retraining_lr': FLAGS.lr,
                            'outer_lr': TRAINING_FLAGS['outer_lr'], 'lr': TRAINING_FLAGS['lr'], 'inner_lr': TRAINING_FLAGS['inner_lr']}
            name = get_quant_config_name()
            yaml.dump(metrics_dict, open(os.path.join(exp_folder, image_name, 'metrics_' + name + '.yml'), 'w'))
            torch.save(model.state_dict(),
                       os.path.join(exp_folder, image_name, 'model_' + name + '.pth'))

if __name__ == '__main__':
    app.run(main)
