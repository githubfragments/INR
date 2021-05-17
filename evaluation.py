import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import ljpeg
import random
import tensorflow.compat.v1 as tf


def mse_func(a,b):
    return np.mean((np.array(a, dtype='int32') - np.array(b, dtype='int32'))**2)
image_dir = "KODAK/*.png"
#image_dir = "test_imgs/test_image_2.jpg"
#image_dir = "/Users/yannickstrumpler/styannic@login.leonhard.ethz.ch/DL_Project/Data/CUB_200_2011/CUB_200_2011/images/007.Parakeet_Auklet/*.jpg"
#image_dir = "/Users/yannickstrumpler/Downloads/Bilder_Afrika/*/*.JPG"
#image_dir = "DIV2K_valid_HR/*.png"
image_glob = glob.glob(image_dir)
#image_dir = glob.glob("test_imgs/test_image_2.jpg")
bpp_list = []
mse_list = []
lpips_list = []

if len(image_glob) > 40:
    image_glob = random.sample(image_glob, 40)
for file in image_glob:
    img = Image.open(file)
    #img = img.crop((0,0,512,512))
    #img = img.crop((0,0,1352, 1352))
    bpp = []
    mse = []
    lpips =[]
    with tf.Session() as sess:
        for quality in range(10,110,10):

            out = io.BytesIO()
            img.save(out, format='jpeg', quality = quality, subsampling=0)
            img.save("jpeg_ref/" + file.split('/')[1].split('.')[0] +"_Q"+ str(quality) + ".jpg", format='jpeg', quality=quality,
                     subsampling=0)
            b = out.tell()
            bpp.append(out.tell() * 8 / (img.size[0] * img.size[1]))

            img_tilde = Image.open(out)
            mse.append(mse_func(img, img_tilde))
            #lpips_out = lpips_tf.lpips(np.array(img, dtype='float32'), np.array(img_tilde, dtype='float32'))
            #lpips.append(sess.run(lpips_out))
            #print(lpips)
        bpp_list.append(bpp)
        mse_list.append(mse)
        #lpips_list.append(lpips)

bpp_avg = np.mean(np.array(bpp_list), axis=0)
mse_avg = np.mean(np.array(mse_list), axis=0)
lpips_avg = np.mean(np.array(lpips_list), axis=0)
#for bpp, mse in zip(bpp_avg, mse_avg):
#   plt.plot(bpp, mse, 'b+')
for bpp, lpip in zip(bpp_avg, mse_avg):
   plt.plot(bpp, lpip, 'b+')

bpp_list_tf = []
mse_list_tf = []
lpips_list_tf = []

lmdas = ['0005', '001', '005','01', '05', '1', '5', '10']

#lmdas = ['0005','01','1', '5']
#lmdas = ['preconv']
#lmdas = ['lpips']
lmdas = ['005']
folder = ['mse_baseline', 'mse_lpips', 'gaussian_lpips', 'add_gaussian']
for f in folder:
    bpp_list_tf = []
    mse_list_tf = []
    lpips_list_tf = []
    for lmda in lmdas:

        #checkpoint = "dense_train/lambda_"+ lmda + "/train"
        #checkpoint = "pre_conv_run2/lambda_"+ lmda + "/train"
        #checkpoint = "sgd_test/lambda_"+ lmda + "/train"
        #checkpoint = "resnet/lambda_" + lmda + "/train"
        #checkpoint = "lpips_test"
        #checkpoint = "lpips_test_run/lambda_" + lmda
        checkpoint = f + "/lambda_" + lmda + "/train"
        bpp_tf, mse_tf, lpips_tf, img_tf = ljpeg.evaluate(image_dir, checkpoint, lmda)
        tf.keras.backend.clear_session()
        bpp_list_tf.append(bpp_tf)
        mse_list_tf.append(mse_tf)
        lpips_list_tf.append(lpips_tf)


    bpp_avg_tf = np.mean(np.array(bpp_list_tf), axis=1)
    mse_avg_tf = np.mean(np.array(mse_list_tf), axis=1)
    lpips_avg_tf = np.mean(np.array(lpips_list_tf), axis=1)
    #for bpp, mse in zip(bpp_avg_tf, mse_avg_tf):
    #   plt.plot(bpp, mse, 'r+')

    for bpp, val in zip(bpp_avg_tf, mse_avg_tf):
       plt.plot(bpp, val, '+', label=f)




#img = Image.open(image_dir[0])
#img_tilde = Image.open("jpeg_eval/test_image_2.jpg")
#mse_tf = mse_func(img.crop((0,0,3032, 4048)), img_tilde)
#plt.plot(bpp_tf, mse_tf, 'g+')

q=[[39, 47, 52, 56, 60, 65, 70, 72, 46, 51, 54, 59, 63, 68, 72, 77, 51, 54, 57, 61, 65, 70, 75, 80, 55, 58, 60, 64, 68, 73, 79, 84, 59, 62, 64, 68, 72, 78, 84, 90, 63, 66, 69, 72, 77, 84, 92, 99, 68, 71, 74, 78, 83, 91, 101, 108, 70, 75, 78, 83, 89, 98, 108, 115], [32, 40, 43, 48, 55, 64, 76, 90, 39, 42, 46, 51, 58, 68, 81, 95, 43, 46, 49, 55, 63, 73, 86, 98, 48, 51, 55, 62, 70, 81, 93, 103, 54, 58, 63, 70, 79, 90, 100, 108, 63, 68, 73, 81, 90, 99, 106, 111, 76, 80, 86, 92, 100, 106, 112, 115, 89, 94, 98, 103, 107, 112, 115, 117]]
#bpp_list = []
#mse_list = []
# for file in image_dir:
#     img = Image.open(file)
#
#     out = io.BytesIO()
#     img.save(out, format='jpeg', qtables=q, subsampling=0)
#     img.save("jpeg_qt/" + file.split('/')[1].split('.')[0] + ".jpg", format='jpeg', qtables=q,
#              subsampling=0)
#     b = out.tell()
#     bpp_list.append(out.tell() * 8 / (img.size[0] * img.size[1]))
#
#     img_tilde = Image.open(out)
#     mse_list.append(mse_func(img, img_tilde))
# for bpp, mse in zip(bpp_list,mse_list):
#    plt.plot(bpp, mse, 'r+')



plt.legend()
plt.show()