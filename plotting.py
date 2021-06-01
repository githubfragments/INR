import pickle
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

with open("plots/mlp_l1_train.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_mlp_l1 = pd.DataFrame.from_dict(in_dict)

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
    #                     c='b', ax=ax, label='MLP')
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


    # df_mlp8.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                      c='y', ax=ax, label='MLP 2 layers (8 frq.)')
    # df_mlp_quant.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                      c='k', ax=ax, label='MLP (7,7,7,6) quant. (8 frq.)')
    # df_log_mse.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                           c='r', ax=ax, label='MLP log mse (8 frq.)')

    df_normal.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                         c='g', ax=ax, label='MLP 4-6 layers (6 frq.)')
    df_mlp_normal.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                           c='b', ax=ax, label='MLP 2 layers (6 frq.)')
    # df_mixture.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
    #                           c='r', ax=ax, label='Mixture (6 frq.)')
    df_mlp_l1_positional.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                            c='r', ax=ax, label='MLP L1 positional')
    df_mlp_l1_gauss.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                                      c='y', ax=ax, label='MLP L1 gauss')
    df_mlp_l1_nerf.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                                 c='m', ax=ax, label='MLP L1 nerf')
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

plt.xlim([0,5.0])
plt.ylim([25,40])
plt.show()