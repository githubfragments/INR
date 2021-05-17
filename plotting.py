import pickle
import pandas as pd
import matplotlib.pyplot as plt


#%%
scale=2
with open("plots/multi_downsampled.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_multi_downsampled = pd.DataFrame.from_dict(in_dict)

with open("plots/multi.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_multi = pd.DataFrame.from_dict(in_dict)

with open("plots/normal.pickle", "rb") as file:
    in_dict = pickle.load(file)
df_normal = pd.DataFrame.from_dict(in_dict)

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
plt.figure()
ax=df_jpeg.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='y', label='JPEG')
df_jpeg_matlab.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='k', label='JPEG Matlab',ax=ax)
df_jpeg2000.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='m', label='JPEG2000 Matlab',ax=ax)
df_bpg.plot(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='c', label='BPG',ax=ax)
df_multi.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='r', label='Multi Scale', ax=ax)
df_multi_downsampled.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='g', label='Multi Scale Downsampled', ax=ax)
df_normal.plot.scatter(x='bpp', y='psnr', xlabel='bpp', ylabel='PSNR',
                    c='b', ax=ax, label='MLP')



plt.xlim([0,3])
plt.ylim([20,45])
plt.show()