#!/bin/bash

#FOLDER_NAME="$1"
#if [ "$1" = "" ]
#then
#    echo "No folder name given"
#    exit
#else
#    mkdir "$FOLDER_NAME"
#fi


#module load gcc/6.3.0 python_gpu/3.7.4
#for i in $(seq 0 0.1 1)
#do
#bsub -oo "$FOLDER_NAME" -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" --exp_root exp --data_root data --model
#done
#echo -e 'all jobs successfully submitted.'
#eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
#source /home/yannick/anaconda3/etc/profile.d/conda.sh
dataset='KODAK21'
#hidden_layers=9
#hidden_dims=100
#for activation in 'sine' 'relu'
#do
#for encoding in  'gauss' 'positional' 'mlp' 'nerf'
#do
#/home/yannick/anaconda3/envs/INR/bin/python trainINR.py --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims
#done
#done
#
#hidden_layers=4
#hidden_dims=200
#for activation in 'sine' 'relu'
#do
#for encoding in 'gauss' 'positional' 'mlp' 'nerf'
#do
#/home/yannick/anaconda3/envs/INR/bin/python   trainINR.py --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims
#done
#done


#hidden_dims=200
#activation='sine'
#for hidden_layers in 2 3 5
#do
#for encoding in 'gauss' 'positional' 'mlp' 'nerf'
#do
#/home/yannick/anaconda3/envs/INR/bin/python   trainINR.py --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims
#done
#done
#for l1_reg in 0.00005
#do
#activation='sine'
#hidden_layers=4
#scale=10
#for encoding in 'positional' 'mlp' 'nerf'
#do
#for hidden_dims in 32 48 64 100
#do
#/home/yannick/anaconda3/envs/INR/bin/python   trainINR.py --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims --encoding_scale $scale --l1_reg $l1_reg
#done
#done
#
#
#activation='sine'
#hidden_layers=4
#encoding='gauss'
#scale=4
#for hidden_dims in 32 48 64 100
#do
#/home/yannick/anaconda3/envs/INR/bin/python   trainINR.py --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims --encoding_scale $scale --l1_reg $l1_reg
#done
#done

#model_type='multi_tapered'
#l1_reg=0.0
#activation='sine'
#for encoding in 'nerf' 'gauss'
#do
#for hidden_layers in 2
#do
#for hidden_dims in 32 48 64 128
#do
#/home/yannick/anaconda3/envs/INR/bin/python   trainINR.py --model_type $model_type --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims  --l1_reg $l1_reg
#done
#done
#done

#for model_type in 'mlp' 'multi_tapered' #'parallel'
#do
#for phase in  '' #--phased'
#do
#intermediate_losses='' #--intermediate_losses'
#l1_reg=0.0
#activation='sine'
#scale=4
#for encoding in  'gauss' 'nerf'
#do
#for hidden_layers in 2
#do
#for hidden_dims in 32 48 64 128
#do
#/home/yannick/anaconda3/envs/INR/bin/python   trainINR.py --model_type $model_type  --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims  --l1_reg $l1_reg --encoding_scale $scale $phase $intermediate_losses
#done
#done
#done
#done
#done

for model_type in 'mlp' #multi' 'multi_tapered' #'parallel'
do
l1_reg=0.0
activation='sine'
scale=4
for encoding in 'nerf'
do
for ff_dims in '8' #'10,8,6' '6,8,10' '8,8,8' '4,6,8' '8,6,4' '6,6,6'
do
for hidden_layers in 2
do
for hidden_dims in 32 48 64 128
do
/home/yannick/anaconda3/envs/INR/bin/python   trainINR.py --exp_root 'exp/exp_log_mse' --loss log_mse --ff_dims $ff_dims --model_type $model_type  --dataset $dataset --activation $activation --encoding $encoding --hidden_layers $hidden_layers --hidden_dims $hidden_dims  --l1_reg $l1_reg --encoding_scale $scale
done
done
done
done
done

cd  ~/trex
./ETH-ethermine.sh





