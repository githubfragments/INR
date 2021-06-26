import pickle

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt


#%%
scale=2
with open("plots/multi_downsampled.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_multi_downsampled = pd.DataFrame.from_dict(in_dict)

with open("plots/multiv2.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_multi = pd.DataFrame.from_dict(in_dict)

with open("plots/normal.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_normal = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp2hlv2.pickle", "rb") as file:
    in_dict = pickle.load(file)
    df_mlp = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_quant_opt.pickle", "rb") as file:
    in_dict = pickle.load(file)

df_mlp_quant = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_log_mse.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_log_mse = pd.DataFrame.from_dict(in_dict)
with open("plots/mixture.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mixture = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_maml_signed_int.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_maml= pd.DataFrame.from_dict(in_dict)

with open("plots/mlp_train_l1reg1e-05_symmetric.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_symmetric = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_l1_train.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1 = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_train_l1reg5e-05_signed_int.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_reg5 = pd.DataFrame.from_dict(in_dict)
#with open("plots/mlp_train_l1reg1e-05_symmetric_adaround_6bit_1hotSTE_v2.pickle", "rb") as file:
#with open("plots/mlp_train_nerf_l1reg1e-05_symmetric_adaround_6bit_1hotSTE_lr1e-07.pickle", "rb") as file:
with open("plots/mlp_train_nerf_l1reg1e-05_symmetric_adaround_6bit_500iter_lr1e-06_1000epochs_weightloss.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_adaround_6bit = pd.DataFrame.from_dict(in_dict)
#with open("plots/mlp_train_nerf_l1reg1e-05_symmetric_adaround_4bit_1hotSTE_lr1e-07.pickle", "rb") as file:
with open("plots/mlp_train_nerf_l1reg1e-05_symmetric_adaround_7bit_500iter_lr1e-06_1000epochs_weightloss.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_adaround_7bit = pd.DataFrame.from_dict(in_dict)
#with open("plots/mlp_train_nerf_l1reg1e-05_symmetric_adaround_5bit_1hotSTE_lr1e-07.pickle", "rb") as file:
with open("plots/mlp_train_nerf_l1reg1e-05_symmetric_adaround_5bit_500iter_lr1e-06_1000epochs_weightloss.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_adaround_5bit = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_train_nerf_l1reg1e-05_symmetric_adaround_8bit_500iter_lr1e-06_1000epochs_weightloss.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_adaround_8bit = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_train_l1reg1e-04_signed_int.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_reg4 = pd.DataFrame.from_dict(in_dict)
with open("plots/mlp_l1_train_and_quant.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1_quant = pd.DataFrame.from_dict(in_dict)


with open("baselines/jpeg_matlab{}x.pickle".format(scale), 'rb') as handle:
    jpeg_matlab_dict = pickle.load(handle)
with open("baselines/jpeg{}x.pickle".format(scale), 'rb') as handle:
    jpeg_dict = pickle.load(handle)
with open("baselines/jpeg_2000_matlab{}x.pickle".format(scale), 'rb') as handle:
        jpeg2000_dict = pickle.load(handle)
with open("baselines/bpg{}x.pickle".format(scale), 'rb') as handle:
    bpg_dict = pickle.load(handle)
df_jpeg = pd.DataFrame.from_dict(jpeg_dict['kodim21'])
df_jpeg_matlab = pd.DataFrame.from_dict(jpeg_matlab_dict['kodim21'])
df_jpeg2000 = pd.DataFrame.from_dict(jpeg2000_dict['kodim21'])
df_bpg = pd.DataFrame.from_dict(bpg_dict['kodim21'])
df_multi= df_multi[df_multi['encoding'] == 'nerf']
df_multi_tapered = df_multi[df_multi['model_type'] == 'multi_tapered']
df_multi= df_multi[df_multi['model_type'] == 'multi']
df_normal = df_normal[df_normal['hidden_layers'] < 8]

df_mlp= df_mlp[df_mlp['encoding'] == 'nerf']
df_mlp_quant= df_mlp_quant[df_mlp_quant['encoding'] == 'nerf']
df_mlp_quant = df_mlp_quant[[(True if d == ['8'] else False) for d in  df_mlp_quant['ff_dims']]]
df_mlp_normal = df_mlp[[(True if (d == None or d == ['4', ''])  else False) for d in  df_mlp['ff_dims']]]
df_mlp8 = df_mlp[[(True if d == ['8'] else False) for d in  df_mlp['ff_dims']]]
# df_mlp4 = df_normal[[(True if d == ['4'] else False) for d in  df_normal['ff_dims']]]
# df_mlp10 = df_normal[[(True if d == ['10'] else False) for d in  df_normal['ff_dims']]]

df_mlp_l1_positional= df_mlp_l1[df_mlp_l1['encoding'] == 'positional']
df_mlp_l1_gauss= df_mlp_l1[df_mlp_l1['encoding'] == 'gauss']
df_mlp_l1_nerf= df_mlp_l1[df_mlp_l1['encoding'] == 'nerf']
df_mlp_l1_symmetric_nerf= df_mlp_l1_symmetric[df_mlp_l1_symmetric['encoding'] == 'nerf']
df_mlp_l1_quant_nerf= df_mlp_l1_quant[df_mlp_l1_quant['encoding'] == 'nerf']
df_mixture = df_mixture[df_mixture['hidden_layers'] == 2]
df_mlp_l1_adaround_6bit_nerf = df_mlp_l1_adaround_6bit[df_mlp_l1_adaround_6bit['encoding'] == 'nerf']
df_mlp_l1_adaround_7bit_nerf = df_mlp_l1_adaround_7bit[df_mlp_l1_adaround_7bit['encoding'] == 'nerf']
df_mlp_l1_adaround_5bit_nerf = df_mlp_l1_adaround_5bit[df_mlp_l1_adaround_5bit['encoding'] == 'nerf']
df_mlp_l1_adaround_8bit_nerf = df_mlp_l1_adaround_8bit[df_mlp_l1_adaround_8bit['encoding'] == 'nerf']
df_maml_noreg = df_maml[df_maml['l1_reg'] == 0]
plt.figure(figsize=[100,80])


ax=df_jpeg.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='y', label='JPEG')
df_jpeg_matlab.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='k', label='JPEG Matlab',ax=ax)
df_jpeg2000.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='m', label='JPEG2000 Matlab',ax=ax)
df_bpg.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='c', label='BPG',ax=ax)
if 1:
    # df_multi.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                   c='r', label='Multi Scale', ax=ax)
    # df_multi_tapered.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                   c='darkorange', label='Multi Scale Tapered', ax=ax)
    # df_multi_downsampled.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                     c='g', label='Multi Scale Downsampled', ax=ax)
    # df_mlp8.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                     c='g', ax=ax, label='MLP')
    # df_mlp4.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                     c='c', ax=ax, label='MLP')
    # df_mlp10.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                     c='g', ax=ax, label='MLP')

    # df_normal.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                     c='g', ax=ax, label='MLP 4 layers (6 frq.)')
    # df_mlp_normal.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                     c='darkviolet', ax=ax, label='MLP 2 layers (6 frq.)')

    # df_mlp.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                      c='g', ax=ax, label='MLP 2 layers (2-14) freq)')



    # df_mlp_quant.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                      c='k', ax=ax, label='MLP (7,7,7,6) quant. (8 frq.)')
    # df_log_mse.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                           c='r', ax=ax, label='MLP log mse (8 frq.)')

    # df_normal.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                      c='g', ax=ax, label='MLP 4-6 layers (6 frq.)')

    # df_mixture.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                           c='hidden_dims', ax=ax, label='Mixture (6 frq.)', colormap='jet')
    # df_mlp_l1_positional.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                         c='r', ax=ax, label='MLP L1 positional')
    # df_mlp_l1_gauss.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                                   c='y', ax=ax, label='MLP L1 gauss')
    # df_mlp_l1_quant_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                                   c='b', ax=ax, label='MLP L1 nerf L1 retrain')
    df_mlp_l1_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                                 c='m', ax=ax, label='MLP L1 nerf')
    # df_mlp_l1_symmetric_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                             c='g', ax=ax, label='MLP L1 symmetric nerf')
    # df_mlp_l1_reg5.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                             c='b', ax=ax, label='MLP L1 nerf reg5e-05')
    # df_mlp_l1_reg4.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                             c='r', ax=ax, label='MLP L1 nerf reg1e-04')
    df_mlp_l1_adaround_8bit_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                                              c='k', ax=ax, label='+ Adaround 8 bit')
    df_mlp_l1_adaround_7bit_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                                              c='darkorange', ax=ax, label='+ Adaround 7 bit')
    df_mlp_l1_adaround_6bit_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                                c='b', ax=ax, label='+ Adaround 6 bit')
    df_mlp_l1_adaround_5bit_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                                              c='g', ax=ax, label='+ Adaround 5 bit')




    # df_mlp8.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR', marker='x',
    #                      c='g', ax=ax, label='MLP 2 layers (8 frq.)')
    # df_mlp_normal.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                            c='k', ax=ax, label='MLP 2 layers (6 frq.)')
    # df_maml.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                             c='l1_reg', ax=ax, label='MAML', colormap='jet', norm=matplotlib.colors.LogNorm())
    # df_maml_noreg.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                      c='darkgrey', ax=ax, label='MAML no reg')
from itertools import cycle
cycol = cycle('bgrcmk')
if False:
    for ff_dims in [['10','8','6'], ['6','8','10'], ['8','8','8'], ['4','6','8'], ['8','6','4'], ['6','6','6']]:
        df_plot = df_multi_tapered[[(True if d == ff_dims else False) for d in  df_multi_tapered['ff_dims']]]
        df_plot.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                              label=str(ff_dims), ax=ax, c=next(cycol))
    for ff_dims in [['2', '4', '8'], ['8', '4', '2'], ['4', '4', '4']]:
        df_plot = df_multi_tapered[[(True if d == ff_dims else False) for d in df_multi_tapered['ff_dims']]]
        df_plot.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                             label=str(ff_dims),marker='x', ax=ax, c=next(cycol))

plt.xlim([0,1.6])
plt.ylim([24,36])
plt.show()