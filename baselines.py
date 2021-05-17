
import PIL.Image
import io
import skimage.measure
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
def mse_func(a,b):
    return np.mean((np.array(a, dtype='float32') - np.array(b, dtype='float32'))**2)
scale = 2

jpeg_dict = {}
imglob = glob.glob('/home/yannick/KODAK/kodim*.png')
jpeg = True
jpeg2000 = True
bpg = True
if jpeg:
    try:
        with open("baselines/jpeg{}x.pickle".format(scale), 'rb') as handle:
            jpeg_dict = pickle.load(handle)
    except:
        for im in imglob:
            img = PIL.Image.open(im)
            img = img.resize((img.size[0]// scale, img.size[1]// scale))
            image_name = im.split('/')[-1].split('.')[0]
            bpp = []
            psnr = []
            lpips = []
            ssim = []
            qualities = list(range(1, 100, 1))
            for quality in qualities:
                out = io.BytesIO()
                img.save(out, format='jpeg', quality=quality, subsampling=0)

                bpp_val = out.tell() * 8 / (img.size[0] * img.size[1])
                bpp.append(bpp_val)

                img_tilde = PIL.Image.open(out)
                #psnr_val = 10 * np.log10(256 ** 2 / mse_func(img, img_tilde))
                psnr_val = skimage.measure.compare_psnr(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'), data_range=256)
                ssim_val = skimage.measure.compare_ssim(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'),
                                                 multichannel=True, data_range=256)
                psnr.append(psnr_val)
                ssim.append(ssim_val)
                # img.save("jpeg_ref/" + im.split('/')[-1].split('.')[0] + "_Q" + str(quality) + '_bpp' + str(bpp_val) + 'psnr' + str(psnr_val) + ".jpg", format='jpeg',
                #          quality=quality, subsampling=0)
                # lpips_out = lpips_tf.lpips(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'))
                # lpips.append(sess.run(lpips_out))
                # print(lpips)
            jpeg_dict[image_name] = {'quality': qualities, 'bpp': bpp, 'psnr': psnr, 'ssim': ssim}

        with open("baselines/jpeg{}x.pickle".format(scale), 'wb') as handle:
            pickle.dump(jpeg_dict, handle)

if jpeg2000:
    jpeg2000_dict = {}
    try:
        with open("baselines/jpeg_2000{}x.pickle".format(scale), 'rb') as handle:
            jpeg2000_dict = pickle.load(handle)
    except:
        for im in imglob:
            img = PIL.Image.open(im)
            img = img.resize((img.size[0] // scale, img.size[1] // scale))
            image_name = im.split('/')[-1].split('.')[0]
            bpp = []
            psnr = []
            lpips = []
            ssim = []
            qualities = list(range(15, 50, 1))
            for quality in qualities:
                out = io.BytesIO()
                img.save(out, format='jpeg2000', quality_layers=[quality], quality_mode='dB')

                bpp_val = out.tell() * 8 / (img.size[0] * img.size[1])
                bpp.append(bpp_val)

                img_tilde = PIL.Image.open(out)
                #psnr_val = 10 * np.log10(256 ** 2 / mse_func(img, img_tilde))
                psnr_val = skimage.measure.compare_psnr(np.array(img, dtype='float32'),
                                                        np.array(img_tilde, dtype='float32'), data_range=256)
                ssim_val = skimage.measure.compare_ssim(np.array(img, dtype='float32'),
                                                        np.array(img_tilde, dtype='float32'),
                                                        multichannel=True, data_range=256)
                psnr.append(psnr_val)
                ssim.append(ssim_val)
                # img.save("jpeg_ref/" + im.split('/')[-1].split('.')[0] + "_Q" + str(quality) + '_bpp' + str(
                #     bpp_val) + 'psnr' + str(psnr_val) + ".jpg", format='jpeg',
                #          quality=quality, subsampling=0)
                # lpips_out = lpips_tf.lpips(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'))
                # lpips.append(sess.run(lpips_out))
                # print(lpips)
            jpeg2000_dict[image_name] = {'quality': qualities, 'bpp': bpp, 'psnr': psnr, 'ssim': ssim}

        with open("baselines/jpeg_2000{}x.pickle".format(scale), 'wb') as handle:
            pickle.dump(jpeg2000_dict, handle)

if bpg:
    imglob = glob.glob("/home/yannick/KODAK{}x/kodim*.png".format(scale))
    bpg_dict = {}
    try:
        with open("baselines/bpg{}x.pickle".format(scale), 'rb') as handle:
            bpg_dict = pickle.load(handle)
    except:
        for im in imglob:
            img = PIL.Image.open(im)
            image_name = im.split('/')[-1].split('.')[0]
            bpp = []
            psnr = []
            lpips = []
            ssim = []
            qualities = list(range(25, 52, 1))
            for quality in qualities:
                bpg_path = "bpg_ref/{}_bpg_Q{}.bpg".format(image_name, quality)
                png_path = "bpg_ref/{}_bpg_Q{}.png".format(image_name, quality)
                cmd = ["/home/yannick/bpg/libbpg-0.9.5/bpgenc", str(im), "-m", "9", "-f", "444", "-o", bpg_path
                       , "-q", str(quality)]
                subprocess.run(cmd)
                cmd = ["/home/yannick/bpg/libbpg-0.9.5/bpgdec", bpg_path, "-o", png_path]
                subprocess.run(cmd)
                num_bytes = Path(bpg_path).stat().st_size
                bpp_val = num_bytes * 8 / (img.size[0] * img.size[1])
                bpp.append(bpp_val)

                img_tilde = PIL.Image.open(png_path)
                #psnr_val = 10 * np.log10(256 ** 2 / mse_func(img, img_tilde))
                psnr_val = skimage.measure.compare_psnr(np.array(img, dtype='float32'),
                                                        np.array(img_tilde, dtype='float32'), data_range=256)
                ssim_val = skimage.measure.compare_ssim(np.array(img, dtype='float32'),
                                                        np.array(img_tilde, dtype='float32'),
                                                        multichannel=True, data_range=256)
                psnr.append(psnr_val)
                ssim.append(ssim_val)
                # img.save("jpeg_ref/" + im.split('/')[-1].split('.')[0] + "_Q" + str(quality) + '_bpp' + str(
                #     bpp_val) + 'psnr' + str(psnr_val) + ".jpg", format='jpeg',
                #          quality=quality, subsampling=0)
                # lpips_out = lpips_tf.lpips(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'))
                # lpips.append(sess.run(lpips_out))
                # print(lpips)

            qualities = bpg_dict[image_name]['quality'] + qualities
            bpp = bpg_dict[image_name]['bpp'] + bpp
            psnr = bpg_dict[image_name]['psnr'] + psnr
            ssim = bpg_dict[image_name]['ssim'] + ssim
            bpg_dict[image_name] = {'quality': qualities, 'bpp': bpp, 'psnr': psnr, 'ssim': ssim}

        with open("baselines/bpg{}x.pickle".format(scale), 'wb') as handle:
            pickle.dump(bpg_dict, handle)


plt.plot(jpeg_dict['kodim21']['bpp'], jpeg_dict['kodim21']['psnr'], label='JPEG')
plt.plot(jpeg2000_dict['kodim21']['bpp'], jpeg2000_dict['kodim21']['psnr'], label='JPEG2000')
plt.plot(bpg_dict['kodim21']['bpp'], bpg_dict['kodim21']['psnr'], label='BPG')
plt.legend()
plt.show()