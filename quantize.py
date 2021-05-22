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

from tqdm import tqdm

#from Quantization import convert_to_nn_module
# Quantization related import
from aimet_torch.quantsim import QuantizationSimModel
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

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        for epoch in range(epochs):
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

                    if train_loss < best_mse:
                        best_state_dict = model.state_dict()

                    optim.zero_grad()
                    scaler.scale(train_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    pbar.update(1)

    model.load_state_dict(best_state_dict, strict=True)
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


def quantize_model(model, bitwidth=8, layerwise_bitwidth=None, retrain=True):
    res = check_metrics(dataloader, model, image_resolution)
    print(res)
    input_shape = coord_dataset.mgrid.shape
    dummy_in = torch.rand(input_shape)
    sim = QuantizationSimModel(model, default_output_bw=31, default_param_bw=bitwidth, dummy_input=dummy_in.cuda())#,
                               # config_file='quantsim_config/'
                               #             'default_config.json')
    modules_to_exclude = (Sine, ImageDownsampling, PosEncodingNeRF, FourierFeatureEncodingPositional, FourierFeatureEncodingGaussian)
    excl_layers = []
    for mod in sim.model.modules():
        if isinstance(mod, QcPostTrainingWrapper) and isinstance(mod._module_to_wrap, modules_to_exclude):
            excl_layers.append(mod)

    sim.exclude_layers_from_quantization(excl_layers)
    i=0
    for name, mod in sim.model.named_modules():
        if isinstance(mod, QcPostTrainingWrapper):
            mod.output_quantizer.enabled = False
            mod.input_quantizer.enabled = False

            if torch.count_nonzero(mod._module_to_wrap.bias.data):
                mod.param_quantizers['bias'].enabled = True
            if layerwise_bitwidth:
                mod.param_quantizers['bias'].bitwidth = layerwise_bitwidth[i]
                mod.param_quantizers['weight'].bitwidth = layerwise_bitwidth[i]
                i += 1
    # Quantize the untrained MNIST model
    sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=5)

    # torch.save(sim.model,
    #            os.path.join(
    #                os.path.join(exp_folder,
    #                             image_name + '/checkpoints/model_aimet_quantized.pth')))
    loss_fn = partial(loss_functions.image_mse, None)
    quantized_model = sim.model
    res = check_metrics(dataloader, quantized_model, image_resolution)
    print(res)

    if retrain:
        quantized_model = retrain_model(sim.model, dataloader, 200, loss_fn, 0.0000005,0)# TRAINING_FLAGS['l1_reg'])
        # Fine-tune the model's parameter using training
        # torch.save(quantized_model,
        #            os.path.join(
        #                os.path.join(exp_folder,
        #                             image_name + '/checkpoints/model_aimet_quantized_retrained.pth')))
        res = check_metrics(dataloader, quantized_model, image_resolution)
   # w = sim.model.net.net[0][0]._module_to_wrap.weight
   # q = sim.model.net.net[0][0].param_quantizers['weight']
   # wq = q.quantize(w, q.round_mode)
    quantized_dict = {}
    for name, module in sim.model.named_modules():
        if isinstance(module, QcPostTrainingWrapper) and isinstance(module._module_to_wrap, torch.nn.Linear):
            weight_quantizer = module.param_quantizers['weight']
            bias_quantizer = module.param_quantizers['bias']

            wrapped_linear = module._module_to_wrap
            weight = wrapped_linear.weight
            bias = wrapped_linear.bias
            quantized_weight = weight_quantizer.quantize(weight, weight_quantizer.round_mode).cpu().detach().numpy() #+ weight_quantizer.encoding.offset
            quantized_bias = bias_quantizer.quantize(bias, bias_quantizer.round_mode).cpu().detach().numpy() #+ bias_quantizer.encoding.offset
            weights_csc = scipy.sparse.csc_matrix(quantized_weight + weight_quantizer.encoding.offset)
            quantized_dict[name] = {'weight': {'data': quantized_weight, 'encoding': weight_quantizer.encoding}, 'bias': {'data': quantized_bias, 'encoding': bias_quantizer.encoding}}
    weights_np = []
    for l in quantized_dict.values():
        w = l['weight']['data']
        b = l['bias']['data']
        Q = l['weight']['encoding'].bw
        if Q < 9:
            tpe = 'uint8'
        elif Q < 17:
            tpe = 'uint16'
        else:
            tpe = 'uint32'
        w = w.astype(tpe).flatten()
        weights_np.append(w)

        if l['bias']['encoding']:
            Q = l['bias']['encoding'].bw
            if Q < 9:
                tpe = 'uint8'
            elif Q < 17:
                tpe = 'uint16'
            else:
                tpe = 'uint32'
            b = b.astype(tpe).flatten()
            weights_np.append(b)
    weights_np = np.concatenate(weights_np)
    comp = zlib.compress(weights_np, level=9)
    print(len(comp))
    # sim.export(path=os.path.join(
    #                os.path.join(exp_folder,
    #                             image_name, 'checkpoints')), filename_prefix='model_aimet_quantized_retrained', dummy_input=dummy_in, set_onnx_layer_names=False)

    print(res)
    return quantized_model, res, len(comp)


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

    #quantize_model(mnist_torch_model.train)
    df.plot.scatter(x='bytes', y='psnr')
    import matplotlib.pyplot as plt
    plt.show()