   
   

# python Trainer.py \
# --tfr 1.0 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 1

#16.896 
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0008_Cyclical_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0008_Cyclical_tfr_d_step1.0_weight/PSNR_40.09970179029331_epoch_69.ckpt


# python Trainer.py \
# --tfr 1.0 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 1

#17.730
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0238_Monotonic_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0238_Monotonic_tfr_d_step1.0_weight/PSNR_40.268_epoch_69.ckpt


# python Trainer.py \
# --tfr 1.0 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type None \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 1

#16.537
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0508_None_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0508_None_tfr_d_step1.0_weight/PSNR_40.324_epoch_69.ckpt


# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.25

#17.730
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0238_Monotonic_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_0238_Monotonic_tfr_d_step1.0_weight/PSNR_40.268_epoch_69.ckpt

#######################################
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.5

# 21.454
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1406_Cyclical_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1406_Cyclical_tfr_d_step1.0_weight/PSNR_22.060_epoch_3.ckpt

# #19.626
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1406_Cyclical_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1406_Cyclical_tfr_d_step1.0_weight/PSNR_21.515_epoch_35.ckpt


# #20.486
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1406_Cyclical_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1406_Cyclical_tfr_d_step1.0_weight/PSNR_21.842_epoch_18.ckpt




# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.5



#21.4
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1552_Monotonic_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1552_Monotonic_tfr_d_step1.0_weight/PSNR_22.057_epoch_6.ckpt


# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.5 \
# --kl_anneal_start_epoch 10

# 19.762
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1855_Cyclical_tfr_d_step1_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230810_1855_Cyclical_tfr_d_step1_weight/PSNR_20.980_epoch_50.ckpt


###########################################
#Late beta
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.5
# --kl_anneal_start_epoch 10

#21.486  (Cyclical)
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_0051_None_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_0051_None_tfr_d_step1.0_weight/PSNR_22.329_epoch_4.ckpt

#21.439  (Cyclical)
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_0051_None_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_0051_None_tfr_d_step1.0_weight/PSNR_21.989_epoch_5.ckpt


# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.5 \
# --kl_anneal_start_epoch 10

# 20.920
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1038_Monotonic_tfr_d_step1.0_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1038_Monotonic_tfr_d_step1.0_weight/PSNR_23.833_epoch_62.ckpt



# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type None \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.5 \
# --kl_anneal_start_epoch 10

# 20.370
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1308_None_tfr_d_step1.0KLratio_0.5_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1308_None_tfr_d_step1.0KLratio_0.5_weight/PSNR_22.460_epoch_61.ckpt

###########################
#0.25
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.25 \
# --kl_anneal_start_epoch 10

#  20.138
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1538_Cyclical_tfr_d_step1.0KLratio_0.25_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1538_Cyclical_tfr_d_step1.0KLratio_0.25_weight/PSNR_22.529_epoch_50.ckpt


# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.25 \
# --kl_anneal_start_epoch 10

# 20.038
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1808_Monotonic_tfr_d_step1.0KLratio_0.25_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230811_1808_Monotonic_tfr_d_step1.0KLratio_0.25_weight/PSNR_21.962_epoch_61.ckpt


######################################
#0.75
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10

#21.248
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0000_Cyclical_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0000_Cyclical_tfr_d_step1.0KLratio_0.75_weight/PSNR_22.841_epoch_65.ckpt



# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10

# 21.884
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0229_Monotonic_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0229_Monotonic_tfr_d_step1.0KLratio_0.75_weight/PSNR_22.599_epoch_31.ckpt



# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type None \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10

# 20.393
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0459_None_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0459_None_tfr_d_step1.0KLratio_0.75_weight/PSNR_23.482_epoch_45.ckpt


######################################
#kl_anneal_cycle
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 5 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10

# 20.759
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0728_Cyclical_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0728_Cyclical_tfr_d_step1.0KLratio_0.75_weight/PSNR_22.874_epoch_61.ckpt


# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 3 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10


#  21.104
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0957_Cyclical_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_0957_Cyclical_tfr_d_step1.0KLratio_0.75_weight/PSNR_23.624_epoch_61.ckpt


# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 6 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10

# 21.405
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_1227_Cyclical_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230812_1227_Cyclical_tfr_d_step1.0KLratio_0.75_weight/PSNR_22.172_epoch_4.ckpt


#####################################
#start
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 5

# 18.382
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0113_Monotonic_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0113_Monotonic_tfr_d_step1.0KLratio_0.75_weight/PSNR_21.430_epoch_4.ckpt


# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 15

# 19.885
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0343_Monotonic_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0343_Monotonic_tfr_d_step1.0KLratio_0.75_weight/PSNR_21.630_epoch_13.ckpt


#####################################
#tfr_d_step
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 2 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10


# 19.944
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0612_Monotonic_tfr_d_step2.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0612_Monotonic_tfr_d_step2.0KLratio_0.75_weight/PSNR_22.081_epoch_22.ckpt



#tfr_d_step
# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 4 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10

# 17.847
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0842_Monotonic_tfr_d_step4.0KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_0842_Monotonic_tfr_d_step4.0KLratio_0.75_weight/PSNR_22.543_epoch_67.ckpt



# python Trainer.py \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 0.5 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75
# --kl_anneal_start_epoch 10

# 21.172
# python Tester.py \
# --save_root /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_1111_Monotonic_tfr_d_step0.5KLratio_0.75_weight \
# --ckpt_path /media/yclin/3TBNAS/DLP/Lab4/LAB4/20230813_1111_Monotonic_tfr_d_step0.5KLratio_0.75_weight/PSNR_21.478_epoch_21.ckpt

############################################
# python Trainer.py \
# --optim AdamW \
# --batch_size 4 \
# --num_epoch 200 \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Monotonic \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75 \
# --kl_anneal_start_epoch 10


# python Tester.py \
# --save_root /media/bspubuntu/3TBNAS/DLP/Lab4/LAB4/20230814_0129_Monotonic_tfr_d_step1.0KLratio_0.75_weight \
# --ckpt_path /media/bspubuntu/3TBNAS/DLP/Lab4/LAB4/20230814_0129_Monotonic_tfr_d_step1.0KLratio_0.75_weight/PSNR_24.585_epoch_100.ckpt


#24.72
# python Trainer.py \
# --optim AdamW \
# --batch_size 4 \
# --num_epoch 200 \
# --tfr 1 \
# --tfr_sde 2 \
# --tfr_d_step 1 \
# --kl_anneal_type Cyclical \
# --kl_anneal_cycle 4 \
# --kl_anneal_ratio 0.75 \
# --kl_anneal_start_epoch 10


python Tester.py \
--save_root /media/bspubuntu/3TBNAS/DLP/Lab4/LAB4/20230814_1738_Cyclical_tfr_d_step1.0KLratio_0.75_weight \
--ckpt_path /media/bspubuntu/3TBNAS/DLP/Lab4/LAB4/20230814_1738_Cyclical_tfr_d_step1.0KLratio_0.75_weight/PSNR_27.256_epoch_196.ckpt


