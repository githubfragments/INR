import copy
import glob
import os
import zlib
from decimal import Decimal
from itertools import combinations, permutations, product

import numpy as np
import scipy
import skimage
import PIL
import torch
import pandas

from modules import Sine, ImageDownsampling, PosEncodingNeRF, FourierFeatureEncodingPositional, FourierFeatureEncodingGaussian
import modules

# Compression-related imports
import yaml
from aimet_common.defs import CostMetric, CompressionScheme, GreedySelectionParameters, RankSelectScheme, TarRankSelectionParameters
from aimet_torch.defs import WeightSvdParameters, SpatialSvdParameters, ChannelPruningParameters, \
    ModuleCompRatioPair
from aimet_torch.compress import ModelCompressor
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.save_utils import SaveUtils
from aimet_common.defs import QuantScheme
from aimet_torch.meta import connectedgraph_utils
from tqdm import tqdm

#from Quantization import convert_to_nn_module
# Quantization related import
from aimet_torch.quantsim import QuantizationSimModel, QuantParams
from aimet_torch import bias_correction
from functools import partial
from siren import loss_functions, dataio
# Both compression and quantization related imports
from aimet_torch.examples import mnist_torch_model
from torch.utils.data import DataLoader
from losses import model_l1
from aimet_torch.qc_quantize_op import QcPostTrainingWrapper
from utils import check_metrics, check_metrics_full, convert_to_nn_module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#exp_folder = 'siren/exp/KODAK21_epochs10000_lr0.0001_hdims100_hlayer4_gauss_sine_enc_scale4.0/'
#TRAINING_FLAGS = yaml.safe_load(open(os.path.join(exp_folder, 'FLAGS.yml'), 'r'))
image_name = 'kodim21'
imglob = glob.glob('/home/yannick/KODAK/kodim21.png')
for im in imglob:
    image_name = im.split('/')[-1].split('.')[0]

    img_dataset = dataio.ImageFile(im)
    img = PIL.Image.open(im)
    scale = 2#TRAINING_FLAGS['downscaling_factor']
    image_resolution = (img.size[1] // scale, img.size[0] // scale)

    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

def evaluate_model(model: torch.nn.Module, eval_iterations: int, use_cuda: bool = False) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param model: Model to evaluate
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    """
    mse, ssim, psnr = check_metrics(dataloader, model, image_resolution)

    return psnr



def retrain_model(model, train_dataloader, epochs, loss_fn, lr, l1_reg):
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
            r = torch.rand(N)

            # for i, q in enumerate(q_wrapper_list):
            #     if r[i] > 0.2:
            #         q.param_quantizers['weight'].enabled = True
            #         q.param_quantizers['bias'].enabled = True
            #     else:
            #         q.param_quantizers['weight'].enabled = False
            #         q.param_quantizers['bias'].enabled = False
            # for i, q in enumerate(q_wrapper_list):
            #     q.param_quantizers['weight'].enabled = True
            #     q.param_quantizers['bias'].enabled = True
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
                    # for i, q in enumerate(q_wrapper_list):
                    #     weight_quantizer = q.param_quantizers['weight']
                    #     bias_quantizer = q.param_quantizers['bias']
                    #     # q.param_quantizers['weight'].enabled = True
                    #     # q.param_quantizers['bias'].enabled = True
                    #     wrapped_linear = q._module_to_wrap
                    #     weight = copy.deepcopy(wrapped_linear.weight)
                    #     bias = copy.deepcopy(wrapped_linear.bias)
                    #
                    #     weight_dequant = weight_quantizer.quantize_dequantize(weight, weight_quantizer.round_mode)
                    #     bias_dequant = bias_quantizer.quantize_dequantize(bias, bias_quantizer.round_mode)
                    #
                    #     # q.param_quantizers['weight'].enabled = False
                    #     # q.param_quantizers['bias'].enabled = False
                    #
                    #     weight_diff = torch.abs(weight_dequant - wrapped_linear.weight)
                    #     bias_diff = torch.abs(bias_dequant - wrapped_linear.bias)
                    #
                    #     # weight_remainder = torch.remainder(weight, weight_quantizer.encoding.delta)
                    #     # weight_remainder = torch.min(weight_remainder, weight_quantizer.encoding.delta - weight_remainder)
                    #     # bias_remainder = torch.remainder(bias, bias_quantizer.encoding.delta)
                    #     # bias_remainder = torch.min(bias_remainder,
                    #     #                              bias_quantizer.encoding.delta - bias_remainder)
                    #     weight_loss += torch.sum(weight_diff) + torch.sum(bias_diff)

                    train_loss = train_loss + 0.0001 * weight_loss
                    optim.zero_grad()
                    scaler.scale(train_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    pbar.update(1)

            for i, q in enumerate(q_wrapper_list):
                q.param_quantizers['weight'].enabled = True
                q.param_quantizers['bias'].enabled = True

            # model_output['model_out'] = model(model_input['coords'])
            # losses = loss_fn(model_output, gt)
            # mse = losses['img_loss']
            m = check_metrics(train_dataloader, model, image_resolution)
            mse, ssim, psnr = m
            if mse < best_mse:
                print(best_mse)
                print(psnr)
                #print(weight_loss)
                best_state_dict = copy.deepcopy(model.state_dict())
                best_mse = mse



    model.load_state_dict(best_state_dict, strict=True)
    m = check_metrics(train_dataloader, model, image_resolution)
    return model



def weight_svd_auto_mode(model, comp_ratio=0.8, retrain=False):
    input_shape = coord_dataset.mgrid.shape

    # Specify the necessary parameters

    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(comp_ratio),
                                              num_comp_ratio_candidates=20)
    #tar_params = TarRankSelectionParameters(num_rank_indices=2)
    #rank_select = RankSelectScheme.tar
    rank_select = RankSelectScheme.greedy
    auto_params = WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select,
                                                     select_params=greedy_params,
                                                     )#modules_to_ignore=[model.conv1])

    params = WeightSvdParameters(mode=WeightSvdParameters.Mode.auto,
                                 params=auto_params)

    #Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1,
                                             input_shape=input_shape,
                                             compress_scheme=CompressionScheme.weight_svd,
                                             cost_metric=CostMetric.memory,
                                             parameters=params)

    compressed_model, stats = results
    # torch.save(compressed_model,
    #            os.path.join(os.path.join(exp_folder, image_name + '/checkpoints/model_aimet_' + str(comp_ratio) +'.pth')))
    #print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily
   # print(os.path.join(os.path.join(exp_folder, image_name + '/checkpoints/model_aimet_.pth')))
    #res = check_metrics(dataloader, compressed_model, image_resolution)
    #print(res)
    loss_fn = partial(loss_functions.image_mse, None)
    if retrain:
        compressed_model = retrain_model(compressed_model, dataloader,2000, loss_fn, 0.00005, TRAINING_FLAGS['l1_reg'])
        torch.save(compressed_model,
                   os.path.join(
                       os.path.join(exp_folder, image_name + '/checkpoints/model_aimet_' + str(comp_ratio) + '_retrained.pth')))
        res = check_metrics(dataloader, compressed_model, image_resolution)
        print(res)
    return compressed_model
class AimetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (self.dataset[idx][0]['coords'].unsqueeze(0), self.dataset[idx][1]['img'])



def quantize_model(model, bitwidth=8, layerwise_bitwidth=None, retrain=True, ref_model=None, flags=None, adaround=False, lr=0.00000001):
    res = check_metrics(dataloader, model, image_resolution)
    print(res)
    input_shape = coord_dataset.mgrid.shape
    dummy_in = ((torch.rand(input_shape).unsqueeze(0) * 2) - 1).cuda()
    aimet_dataloader = DataLoader(AimetDataset(coord_dataset), shuffle=True, batch_size=1, pin_memory=True,
                                  num_workers=0)
    # Create QuantSim using adarounded_model
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
    res = check_metrics(dataloader, sim.model, image_resolution)
    print(res)
    if adaround:

        params = AdaroundParameters(data_loader=aimet_dataloader, num_batches=1, default_num_iterations=500,
                                    default_reg_param=0.001, default_beta_range=(20, 2))
        # adarounded_model_1 = Adaround.apply_adaround(model=model, dummy_input=dummy_in, params=params,path='', filename_prefix='adaround',
        #                               default_param_bw=bitwidth, ignore_quant_ops_list=excl_layers )
        # Compute only param encodings
        Adaround._compute_param_encodings(sim)

        # Get the module - activation function pair using ConnectedGraph
        module_act_func_pair = connectedgraph_utils.get_module_act_func_pair(model, dummy_in)

        Adaround._adaround_model(model, sim, module_act_func_pair, params, dummy_in)
        #res = check_metrics(dataloader, sim.model, image_resolution)
        #print('1st stage ada round ', res)
        # Update every module (AdaroundSupportedModules) weight with Adarounded weight (Soft rounding)
        Adaround._update_modules_with_adarounded_weights(sim)
        path=''


        # from aimet_torch.cross_layer_equalization import equalize_model
        # equalize_model(model, input_shape)

        # params = QuantParams(weight_bw=4, act_bw=4, round_mode="nearest", quant_scheme='tf_enhanced')
        #
        # # Perform Bias Correction
        # bias_correction.correct_bias(model.to(device="cuda"), params, num_quant_samples=1,
        #                              data_loader=aimet_dataloader, num_bias_correct_samples=1)

        # torch.save(sim.model,
        #            os.path.join(
        #                os.path.join(exp_folder,
        #                             image_name + '/checkpoints/model_aimet_quantized.pth')))

        quantized_model = sim.model
        #res = check_metrics(dataloader, sim.model, image_resolution)
        #print('After Adaround ', res)
    #
    # if retrain:
    #     loss_fn = partial(loss_functions.image_mse, None)
    #     #quantized_model = retrain_model(sim.model, dataloader, 200, loss_fn, 0.0000005, flags['l1_reg'] if flags is not None else 0)
    #     quantized_model = retrain_model(sim.model, dataloader, 300, loss_fn, lr,
    #                                     flags['l1_reg'] if flags is not None else 0)
    #     # Fine-tune the model's parameter using training
    #     # torch.save(quantized_model,
    #     #            os.path.join(
    #     #                os.path.join(exp_folder,
    #     #                             image_name + '/checkpoints/model_aimet_quantized_retrained.pth')))
    #     res = check_metrics(dataloader, quantized_model, image_resolution)
    #     print('After retraining ',res)
    #     state_dict ={}
    #     quantized_dict = {}
    #     for name, module in sim.model.named_modules():
    #         if isinstance(module, QcPostTrainingWrapper) and isinstance(module._module_to_wrap, torch.nn.Linear):
    #             weight_quantizer = module.param_quantizers['weight']
    #             bias_quantizer = module.param_quantizers['bias']
    #             weight_quantizer.enabled = True
    #             bias_quantizer.enabled = True
    #             weight_quantizer.use_soft_rounding = False
    #             bias_quantizer.use_soft_rounding = False
    #             wrapped_linear = module._module_to_wrap
    #             weight = wrapped_linear.weight
    #             bias = wrapped_linear.bias
    #             if not (torch.all(weight < weight_quantizer.encoding.max) and torch.all(
    #                     weight > weight_quantizer.encoding.min)):
    #                 print("not within bounds")
    #
    #             weight_dequant = weight_quantizer.quantize_dequantize(weight,
    #                                                                                 weight_quantizer.round_mode).cpu().detach()
    #             state_dict[name + '.weight'] = weight_dequant
    #             # assert(len(torch.unique(state_dict[name + '.weight'])) <= 2**bitwidth)
    #             bias_dequant = bias_quantizer.quantize_dequantize(bias,
    #                                                                             bias_quantizer.round_mode).cpu().detach()
    #             state_dict[name + '.bias'] = bias_dequant
    #             # assert (len(torch.unique(state_dict[name + '.bias'])) <= 2 ** bitwidth)
    #             quantized_weight = weight_dequant / weight_quantizer.encoding.delta
    #             quantized_bias = bias_dequant / bias_quantizer.encoding.delta
    #             weights_csc = scipy.sparse.csc_matrix(quantized_weight + weight_quantizer.encoding.offset)
    #             quantized_dict[name] = {'weight': {'data': quantized_weight, 'encoding': weight_quantizer.encoding},
    #                                     'bias': {'data': quantized_bias, 'encoding': bias_quantizer.encoding}}
    #     res = check_metrics(dataloader, quantized_model, image_resolution)
    #     print('After hard rounding ', res)

    if adaround:


        filename_prefix = 'adaround'
        # Export quantization encodings to JSON-formatted file
        Adaround._export_encodings_to_json(path, filename_prefix, sim)
        #res = check_metrics(dataloader, sim.model, image_resolution)
        SaveUtils.remove_quantization_wrappers(sim.model)
        adarounded_model = sim.model

        #print('After Adaround ', res)



        sim = QuantizationSimModel(adarounded_model, default_param_bw=bitwidth,
                                   default_output_bw=31, dummy_input=dummy_in)

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

        sim.set_and_freeze_param_encodings(encoding_path='adaround.encodings')

        # Quantize the untrained MNIST model
        #sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=5)
        res = check_metrics(dataloader, sim.model, image_resolution)
        print(res)
    if retrain:
        loss_fn = partial(loss_functions.image_mse, None)
        #quantized_model = retrain_model(sim.model, dataloader, 200, loss_fn, 0.0000005, flags['l1_reg'] if flags is not None else 0)
        quantized_model = retrain_model(sim.model, dataloader, 1000, loss_fn, lr,
                                        flags['l1_reg'] if flags is not None else 0)
        #sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=5)
        # Fine-tune the model's parameter using training
        # torch.save(quantized_model,
        #            os.path.join(
        #                os.path.join(exp_folder,
        #                             image_name + '/checkpoints/model_aimet_quantized_retrained.pth')))
        res = check_metrics(dataloader, quantized_model, image_resolution)
        print('After retraining ',res)


   # # w = sim.model.net.net[0][0]._module_to_wrap.weight
   # q = sim.model.net.net[0][0].param_quantizers['weight']
   # wq = q.quantize(w, q.round_mode)

    #Compute the difference for each parameter
    if ref_model is not None:
        new_state_dict=sim.model.state_dict()
        lis = [[i, j,  a, b] for i, a in ref_model.named_parameters() for j, b in sim.model.named_parameters() if i == j.replace('._module_to_wrap','')]
        for module in lis:
            new_state_dict[module[1]] = module[3] - module[2]
        sim.model.load_state_dict(new_state_dict)
        #sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=1)

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
            if not (torch.all(weight < weight_quantizer.encoding.max) and torch.all(weight > weight_quantizer.encoding.min)):
                print("not within bounds")

            state_dict[name + '.weight'] = weight_quantizer.quantize_dequantize(weight,weight_quantizer.round_mode).cpu().detach()
            #assert(len(torch.unique(state_dict[name + '.weight'])) <= 2**bitwidth)
            state_dict[name + '.bias'] = bias_quantizer.quantize_dequantize(bias, bias_quantizer.round_mode).cpu().detach()
            #assert (len(torch.unique(state_dict[name + '.bias'])) <= 2 ** bitwidth)
            quantized_weight = weight_quantizer.quantize(weight, weight_quantizer.round_mode).cpu().detach().numpy() + weight_quantizer.encoding.offset
            quantized_bias = bias_quantizer.quantize(bias, bias_quantizer.round_mode).cpu().detach().numpy() + bias_quantizer.encoding.offset
            weights_csc = scipy.sparse.csc_matrix(quantized_weight + weight_quantizer.encoding.offset)
            quantized_dict[name] = {'weight': {'data': quantized_weight, 'encoding': weight_quantizer.encoding}, 'bias': {'data': quantized_bias, 'encoding': bias_quantizer.encoding}}

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
    print(len(comp))
    # sim.export(path=os.path.join(
    #                os.path.join(exp_folder,
    #                             image_name, 'checkpoints')), filename_prefix='model_aimet_quantized_retrained', dummy_input=dummy_in, set_onnx_layer_names=False)

    print(res)
    return quantized_model, res, len(comp), state_dict


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_folder = 'siren/exp/KODAK21_epochs10000_lr0.0001_hdims32_hlayer4_nerf_sine_l1_reg5e-05_enc_scale10.0/'
    #exp_folder = 'siren/exp/KODAK21_epochs10000_lr0.0001_hdims64_hlayer4_nerf_sine_enc_scale10.0/'
    exp_folder = '/home/yannick/PycharmProjects/INR/exp/KODAK21_epochs10000_lr0.0001_mlp_[8]_hdims64_hlayer2_nerf_sine_enc_scale4.0'
    TRAINING_FLAGS = yaml.safe_load(open(os.path.join(exp_folder, 'FLAGS.yml'), 'r'))
    image_name = 'kodim21'
    imglob = glob.glob('/home/yannick/KODAK/kodim21.png')
    for im in imglob:
        image_name = im.split('/')[-1].split('.')[0]

        img_dataset = dataio.ImageFile(im)
        img = PIL.Image.open(im)
        scale = TRAINING_FLAGS['downscaling_factor']
        image_resolution = (img.size[1] // scale, img.size[0] // scale)

        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)

        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
        input_shape = (1, coord_dataset.mgrid.shape[0], coord_dataset.mgrid.shape[1])
        print(input_shape)
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
    model = model.to(device)
        #state_dict = torch.load('siren/experiment_scripts/logs/' + experiment_name + image_name + '/checkpoints/model_current.pth', map_location='cpu')
    state_dict = torch.load(os.path.join(exp_folder, image_name + '/checkpoints/model_best_.pth'), map_location='cpu')

    model.load_state_dict(state_dict, strict=True)
    #path = os.path.join('../', 'data', 'mnist_trained_on_GPU.pth')
    #path = "siren/exp/KODAK21_epochs10000_lr0.0001_hdims32_hlayer4_gauss_sine_enc_scale4.0/kodim21/checkpoints/model_best_.pth"
    # load trained MNIST model
    #model = torch.load(path)
    model = convert_to_nn_module(model)
    #spatial_svd_manual_mode()
    #spatial_svd_auto_mode()

    #weight_svd_manual_mode()
    #model = weight_svd_auto_mode(model)
    df_list = []
    linear_layer_count = sum([isinstance(module, torch.nn.Linear) for module in model.modules()])
    all_combinations = product(range(6,9), repeat=linear_layer_count)
    for comb in all_combinations:
        quantized_model, res, bytes = quantize_model(model, layerwise_bitwidth=comb, retrain=False)
        mse, ssim, psnr = res
        df_list.append({'bitwidths': comb, 'psnr': psnr, 'bytes': bytes})
    df = pandas.DataFrame.from_records(df_list)
   # channel_pruning_manual_mode()
    #channel_pruning_auto_mode()

    # #quantize_model(mnist_torch_model.train)
    # df.plot.scatter(x='bytes', y='psnr')
    # import matplotlib.pyplot as plt
    # plt.show()