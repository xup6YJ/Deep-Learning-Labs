




#50
# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -model_n ResNet50

# python infer.py \
# -model_n ResNet50 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_50_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230731_1951ResNet50_weight/epoch_255_20230731_1951_ResNet50_95.62.pt

# python infer.py \
# -model_n ResNet50 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_50_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230731_1951ResNet50_weight/epoch_255_20230731_1951_ResNet50_95.62.pt \
# -five_crop 5

# python infer.py \
# -model_n ResNet50 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_50_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230731_1951ResNet50_weight/epoch_255_20230731_1951_ResNet50_95.62.pt \
# -five_crop 10



#152
# python main.py \
# -num_epochs 300 \
# -batch_size 16 \
# -lr 0.001 \
# -model_n ResNet152

# python infer.py \
# -model_n ResNet152 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_152_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230801_1032ResNet152_weight/epoch_296_20230801_1032_ResNet152_94.25.pt \
# -five_crop 0

# python infer.py \
# -model_n ResNet152 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_152_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230801_1032ResNet152_weight/epoch_296_20230801_1032_ResNet152_94.25.pt \
# -five_crop 5

# python infer.py \
# -model_n ResNet152 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_152_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230801_1032ResNet152_weight/epoch_296_20230801_1032_ResNet152_94.25.pt \
# -five_crop 10


#18
# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.001 \
# -model_n ResNet18

# python infer.py \
# -model_n ResNet18 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230731_1409ResNet18_weight/20230731_1409_ResNet18_95.81.pt

# python infer.py \
# -model_n ResNet18 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230731_1409ResNet18_weight/20230731_1409_ResNet18_95.81.pt
# -five_crop



###Hist

# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.001 \
# -model_n ResNet18 \
# -hist \
# -in_ch 1

# python infer.py \
# -model_n ResNet18 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230802_1347ResNet18_weight/epoch_241_20230802_1347_ResNet18_86.43.pt

# python infer.py \
# -model_n ResNet50 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230802_1347ResNet18_weight/epoch_241_20230802_1347_ResNet18_86.43.pt \
# -five_crop 5 \
# -hist \
# -in_ch 1

# python infer.py \
# -model_n ResNet50 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230802_1347ResNet18_weight/epoch_241_20230802_1347_ResNet18_86.43.pt \
# -five_crop 10 \
# -hist \
# -in_ch 1


###Lr scheduler

# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.1 \
# -model_n ResNet18 \
# -sch_n cosine


###Center crop
# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.001 \
# -model_n ResNet18 \
# -sch_n plateau

# python infer.py \
# -model_n ResNet18 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230803_2347ResNet18_weight/epoch_295_20230803_2347_ResNet18_96.25.pt

# python infer.py \
# -model_n ResNet50 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230803_2347ResNet18_weight/epoch_295_20230803_2347_ResNet18_96.25.pt \
# -five_crop 5 \


# python infer.py \
# -model_n ResNet50 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230803_2347ResNet18_weight/epoch_295_20230803_2347_ResNet18_96.25.pt \
# -five_crop 10 \


#Crop 384
# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.001 \
# -model_n ResNet18 \
# -sch_n plateau


# python infer.py \
# -model_n ResNet18 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230804_1228ResNet18_weight/epoch_280_20230804_1228_ResNet18_96.37.pt

# python infer.py \
# -model_n ResNet18 \
# -test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
# -model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230804_1228ResNet18_weight/epoch_280_20230804_1228_ResNet18_96.37.pt \
# -five_crop 5 \


python infer.py \
-model_n ResNet18 \
-test_path /media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv \
-model_w_path /media/yclin/3TBNAS/DLP/Lab3/20230804_1228ResNet18_weight/epoch_280_20230804_1228_ResNet18_96.37.pt \
-five_crop 10 \