# Enable import from parent package

# Lint as: python3
"""Training script."""
import copy

import yaml
from absl import app
from absl import flags
import sys
import os

import losses

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append("siren")
import PIL
from siren import dataio, meta_modules, loss_functions
import modules
import siren.utils as siren_utils
import trainingAC
import utils
import glob
import numpy as np
import skimage
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import torch
import json
from losses import image_log_mse
# Define Flags.
flags.DEFINE_string('data_root',
                    '/home/yannick',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')

flags.DEFINE_enum('dataset', 'KODAK21',
                  ['KODAK', 'KODAK21'],
                  'Dataset used during training.')
flags.DEFINE_integer('batch_size',
                     1,
                     'Batch size used during training.',
                     lower_bound=1)
flags.DEFINE_integer('epochs',
                     10000,
                     'Maximum number of epochs.',
                     lower_bound=1)
flags.DEFINE_float('lr',
                   0.0001,
                   'Learning rate used during training.',
                   lower_bound=0.0)
flags.DEFINE_float('l1_reg',
                   0.0,
                   'L1 weight regularization.',
                   lower_bound=0.0)
flags.DEFINE_float('spec_reg',
                   0,
                   'Spectral regularization. Penalizes the largest singular value of affine layers.',
                   lower_bound=0.0)


flags.DEFINE_enum('activation',
                  'sine',
                  ['sine', 'relu'],
                  'Activation Function.')
flags.DEFINE_enum('model_type',
                  'mlp',
                  ['mlp', 'multi', 'multi_tapered', 'parallel', 'mixture'],
                  'Model Architecture.')
flags.DEFINE_enum('loss',
                  'mse',
                  ['mse', 'log_mse'],
                  'Loss function to use.')
flags.DEFINE_bool('l1_diff', False, 'Compute L1 as difference to initialization.')
flags.DEFINE_enum('encoding', 'mlp', ['mlp', 'nerf', 'positional', 'gauss'], 'Input encoding type used')

flags.DEFINE_integer('hidden_dims',
                     100,
                     'Hidden dimension of fully-connected neural network.',
                     lower_bound=1)
flags.DEFINE_integer('hidden_layers',
                     2,
                     'Number of hidden layer of ' +
                     'fully-connected neural network.',
                     lower_bound=1)
flags.DEFINE_integer('downscaling_factor',
                     2,
                     'Factor by which in input is downsampled',
                     lower_bound=1)
flags.DEFINE_list('ff_dims',
                     None,
                     'Number of fourier feature frequencies for input encoding at different scales')
flags.DEFINE_bool('bn', False, 'Enable batch norm after linear layers.')
flags.DEFINE_integer('epochs_til_ckpt', 1000, 'Time interval in seconds until checkpoint is saved.')
flags.DEFINE_integer('steps_til_summary', 1000, 'Time interval in seconds until tensorboard summary is saved.')
flags.DEFINE_float('encoding_scale', 0.0, 'Standard deviation of the encoder')
flags.DEFINE_bool('phased', False, 'Enable phased training for parallel architecture.')
flags.DEFINE_bool('intermediate_losses', False, 'Enable intermediate losses for parallel architecture.')
flags.DEFINE_integer('num_components', 1, 'Number of components used in the mixture of INRs')
FLAGS = flags.FLAGS


def get_experiment_folder():
    """Create string with experiment name and get number of experiment."""

    # create exp folder
    exp_name = '_'.join([
      FLAGS.dataset, #'batch_size' + str(FLAGS.batch_size),
      'epochs' + str(FLAGS.epochs), 'lr' + str(FLAGS.lr)])
    exp_name = '_'.join([exp_name,str(FLAGS.model_type)])
    if FLAGS.model_type == 'mixture':
        exp_name = '_'.join([exp_name, str(FLAGS.num_components)])
    if FLAGS.ff_dims:
        ff_dims = [int(s) for s in FLAGS.ff_dims]
        exp_name = '_'.join([exp_name, str(ff_dims)])
    exp_name = '_'.join([
        exp_name, 'hdims' + str(FLAGS.hidden_dims),
        'hlayer' + str(FLAGS.hidden_layers)
    ])
    exp_name = '_'.join([exp_name, str(FLAGS.encoding)])
    exp_name = '_'.join([exp_name, str(FLAGS.activation)])
    if FLAGS.phased:
        exp_name = '_'.join([exp_name, 'phased'])
    if FLAGS.intermediate_losses:
        exp_name = '_'.join([exp_name, 'intermediate_losses'])
    if FLAGS.bn:
        exp_name = '_'.join([exp_name, 'bn'])
    if FLAGS.l1_reg > 0.0:
        if FLAGS.l1_diff:
            exp_name = '_'.join([exp_name, 'l1_reg_diff' + str(FLAGS.l1_reg)])
        else:
            exp_name = '_'.join([exp_name, 'l1_reg' + str(FLAGS.l1_reg)])
    if FLAGS.spec_reg > 0.0:
        exp_name = '_'.join([exp_name, 'spec_reg' + str(FLAGS.spec_reg)])
    if FLAGS.encoding_scale > 0.0:
        exp_name = '_'.join([exp_name, 'enc_scale' + str(FLAGS.encoding_scale)])


    exp_folder = os.path.join(FLAGS.exp_root, exp_name)

    if not os.path.exists(exp_folder):

        os.makedirs(exp_folder)

    return exp_folder







def check_metrics(test_loader: DataLoader, model: torch.nn.Module, image_resolution):
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            predictions = model.predict(model_input)
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


    return mse, ssim, psnr

def main(_):
    imglob = glob.glob(os.path.join(FLAGS.data_root, FLAGS.dataset, '*'))
    mses = {}
    psnrs = {}
    ssims = {}
    experiment_folder = get_experiment_folder()
    # save FLAGS to yml
    yaml.dump(FLAGS.flag_values_dict(), open(os.path.join(experiment_folder, 'FLAGS.yml'), 'w'))

    for i, im in enumerate(imglob):
        print('Image: ' + str(i))
        image_name = im.split('/')[-1].split('.')[0]
        img_dataset = dataio.ImageFile(im)
        img = PIL.Image.open(im)
        image_resolution = (img.size[1] // FLAGS.downscaling_factor, img.size[0] // FLAGS.downscaling_factor)

        #run_name = image_name + '_layers' + str(layers) + '_units' + str(hidden_units) + '_model' + FLAGS.model_type
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)


        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=FLAGS.batch_size, pin_memory=True, num_workers=0)
        linear_decay = {'img_loss': trainingAC.LinearDecaySchedule(1, 1, FLAGS.epochs)}
        # Define the model.
        if FLAGS.model_type == 'mlp':
            model = modules.SingleBVPNet_INR(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                                         out_features=img_dataset.img_channels, hidden_features=FLAGS.hidden_dims,
                                         num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale,
                                         batch_norm=FLAGS.bn, ff_dims=FLAGS.ff_dims)
        elif FLAGS.model_type == 'multi_tapered':
            model = modules.MultiScale_INR(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                                         out_features=img_dataset.img_channels, hidden_features=FLAGS.hidden_dims,
                                         num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale, tapered=True, downsample=False, ff_dims=FLAGS.ff_dims)
        elif FLAGS.model_type == 'multi':
            model = modules.MultiScale_INR(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                                         out_features=img_dataset.img_channels, hidden_features=FLAGS.hidden_dims,
                                         num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale, tapered=False,  downsample=False, ff_dims=FLAGS.ff_dims)
        elif FLAGS.model_type == 'parallel':
            model = modules.Parallel_INR(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                                         out_features=img_dataset.img_channels, hidden_features=[FLAGS.hidden_dims//4, FLAGS.hidden_dims//2, FLAGS.hidden_dims],
                                         num_hidden_layers=FLAGS.hidden_layers , encoding_scale=FLAGS.encoding_scale)

        elif FLAGS.model_type == 'mixture':
            model = modules.INR_Mixture(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                                         out_features=img_dataset.img_channels, hidden_features=FLAGS.hidden_dims,
                                         num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale,
                                         batch_norm=FLAGS.bn, ff_dims=FLAGS.ff_dims, num_components=FLAGS.num_components)
        model.cuda()
        root_path = os.path.join(experiment_folder,image_name)


        # Define the loss
        if FLAGS.loss == 'mse':
            loss_fn = partial(loss_functions.image_mse, None)
        elif FLAGS.loss == 'log_mse':
            loss_fn = image_log_mse
        summary_fn = partial(siren_utils.write_image_summary, image_resolution)
        l1_loss_fn = losses.model_l1
        if FLAGS.l1_diff:
            ref_model = copy.deepcopy(model)
            l1_loss_fn = partial(losses.model_l1_diff, ref_model)


        if FLAGS.model_type == 'parallel':
            trainingAC.train_phased(model=model, train_dataloader=dataloader, epochs=FLAGS.epochs, lr=FLAGS.lr,
                             steps_til_summary=FLAGS.steps_til_summary, epochs_til_checkpoint=FLAGS.epochs_til_ckpt,
                             model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, l1_reg=FLAGS.l1_reg,
                             spec_reg=FLAGS.spec_reg, phased=FLAGS.phased, intermediate_losses=FLAGS.intermediate_losses)
        else:
            trainingAC.train(model=model, train_dataloader=dataloader, epochs=FLAGS.epochs, lr=FLAGS.lr,
                         steps_til_summary=FLAGS.steps_til_summary, epochs_til_checkpoint=FLAGS.epochs_til_ckpt,
                         model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, l1_reg=FLAGS.l1_reg, l1_loss_fn=l1_loss_fn, spec_reg=FLAGS.spec_reg, loss_schedules=None)

        mse, ssim, psnr = utils.check_metrics_full(dataloader, model, image_resolution)
        mses[image_name] = mse
        psnrs[image_name] = psnr
        ssims[image_name] = ssim
        torch.save(ref_model.state_dict(),
                   os.path.join(root_path, 'model_init.pth'))

    metrics = {'mse': mses, 'psnr': psnrs, 'ssim': ssims}


    with open(os.path.join(experiment_folder, 'result.json'), 'w') as fp:
        json.dump(metrics, fp)

if __name__ == '__main__':
    app.run(main)
