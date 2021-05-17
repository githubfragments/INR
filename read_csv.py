import pandas
import pickle
df = pandas.read_csv('/home/yannick/Downloads/KODAK2x.csv')

df_jpeg_bpp = df[1:25]
df_jpeg_psnr = df[28:52]
df_jpeg2000_bpp = df[56:80]
df_jpeg2000_psnr = df[84:108]




jpeg_dict = {}
jpeg2000_dict = {}
scale = 2
for i in range(1, 25):
    jpeg_bpp = [float(s.replace(',', '.')) for s in df_jpeg_bpp.to_numpy()[i-1, 1:]]
    jpeg_psnr = [float(s.replace(',', '.')) for s in df_jpeg_psnr.to_numpy()[i - 1, 1:]]
    jpeg2000_bpp = [float(s.replace(',', '.')) for s in df_jpeg2000_bpp.to_numpy()[i - 1, 1:-2]]
    jpeg2000_psnr = [float(s.replace(',', '.')) for s in df_jpeg2000_psnr.to_numpy()[i - 1, 1:-2]]

    im_name = f'kodim{i:02}'
    jpeg_dict[im_name] = {'psnr': jpeg_psnr, 'bpp': jpeg_bpp}
    jpeg2000_dict[im_name] = {'psnr': jpeg2000_psnr, 'bpp': jpeg2000_bpp}
    with open("baselines/jpeg_matlab{}x.pickle".format(scale), 'wb') as handle:
        pickle.dump(jpeg_dict, handle)
    with open("baselines/jpeg_2000_matlab{}x.pickle".format(scale), 'wb') as handle:
        pickle.dump(jpeg2000_dict, handle)
