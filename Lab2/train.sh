

# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n EEGNet
# #Accuracy of ReLu: 81.2037037037037, Accuracy of ELu: 82.31481481481482, Accuracy of LeakyReLu: 83.14814814814815


# python main.py \
# -num_epochs 300 \
# -batch_size 16 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n EEGNet
#Test acc of ReLu: 84.35185185185186, Test acc of ELu: 81.11111111111111, Test acc of LeakyReLu: 85.74074074074073

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n EEGNet
#Test acc of ReLu: 86.75925925925925, Test acc of ELu: 82.5, Test acc of LeakyReLu: 84.81481481481481

python main.py \
-num_epochs 300 \
-batch_size 32 \
-lr 0.001 \
-dropout_rate 0.25 \
-elu_alpha 1.0 \
-model_n EEGNet
#sigmoid Test acc of ReLu: 85.92592592592592, Test acc of ELu: 80.83333333333333, Test acc of LeakyReLu: 83.98148148148148
#softmax Test acc of ReLu: 84.81481481481481, Test acc of ELu: 81.38888888888889, Test acc of LeakyReLu: 86.20370370370371

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.2 \
# -elu_alpha 1.0 \
# -model_n EEGNet
#Test acc of ReLu: 86.29629629629629, Test acc of ELu: 83.05555555555556, Test acc of LeakyReLu: 86.85185185185185
#no sig Test acc of ReLu: 85.64814814814815, Test acc of ELu: 84.07407407407408, Test acc of LeakyReLu: 86.38888888888889
#soft Test acc of ReLu: 85.46296296296296, Test acc of ELu: 73.42592592592592, Test acc of LeakyReLu: 87.22222222222223


# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.15 \
# -elu_alpha 1.0 \
# -model_n EEGNet
#Test acc of ReLu: 84.07407407407408, Test acc of ELu: 80.83333333333333, Test acc of LeakyReLu: 82.77777777777777
#no sig Test acc of ReLu: 86.29629629629629, Test acc of ELu: 81.75925925925925, Test acc of LeakyReLu: 84.9074074074074

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n EEGNet
#Test acc of ReLu: 85.55555555555556, Test acc of ELu: 82.31481481481482, Test acc of LeakyReLu: 85.37037037037038

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.1 \
# -elu_alpha 1.0 \
# -model_n EEGNet
#no sig Test acc of ReLu: 83.7037037037037, Test acc of ELu: 81.48148148148148, Test acc of LeakyReLu: 83.88888888888889

#Dropout 
# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n EEGNet

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.5 \
# -elu_alpha 1.0 \
# -model_n EEGNet

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.75 \
# -elu_alpha 1.0 \
# -model_n EEGNet


#ELU
# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 0.4 \
# -model_n EEGNet

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 0.7 \
# -model_n EEGNet

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n EEGNet


########################################

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.2 \
# -elu_alpha 1.0 \
# -model_n DeepConvNet
#soft lr 0.5 30 Test acc of ReLu: 84.35185185185186, Test acc of ELu: 81.66666666666667, Test acc of LeakyReLu: 83.05555555555556
#soft lr 0.1 10 Test acc of ReLu: 82.22222222222221, Test acc of ELu: 80.0925925925926, Test acc of LeakyReLu: 82.5925925925926

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.2 \
# -elu_alpha 0.5 \
# -model_n DeepConvNet
#Test acc of ReLu: 82.87037037037037, Test acc of ELu: 82.96296296296296, Test acc of LeakyReLu: 83.51851851851852

# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.001 \
# -dropout_rate 0.2 \
# -elu_alpha 0.2 \
# -model_n DeepConvNet
#Test acc of ReLu: 83.33333333333334, Test acc of ELu: 83.14814814814815, Test acc of LeakyReLu: 82.96296296296296

########################################

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.2 \
# -elu_alpha 1.0 \
# -model_n AEEGNet
#Test acc of ReLu: 86.85185185185185, Test acc of ELu: 83.14814814814815, Test acc of LeakyReLu: 85.18518518518519

# python main.py \
# -num_epochs 300 \
# -batch_size 64 \
# -lr 0.001 \
# -dropout_rate 0.2 \
# -elu_alpha 1.0 \
# -model_n AEEGNet
#Test acc of ReLu: 86.20370370370371, Test acc of ELu: 83.24074074074073, Test acc of LeakyReLu: 85.64814814814815

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n AEEGNet
#Test acc of ReLu: 86.85185185185185, Test acc of ELu: 81.85185185185185, Test acc of LeakyReLu: 86.94444444444444

# python main.py \
# -num_epochs 300 \
# -batch_size 32 \
# -lr 0.001 \
# -dropout_rate 0.25 \
# -elu_alpha 1.0 \
# -model_n AEEGNet
#first layer attention
#Test acc of ReLu: 86.38888888888889, Test acc of ELu: 82.22222222222221, Test acc of LeakyReLu: 86.94444444444444
#F and S
#Test acc of ReLu: 87.96296296296296, Test acc of ELu: 83.7037037037037, Test acc of LeakyReLu: 86.57407407407408
#Test acc of ReLu: 86.48148148148148, Test acc of ELu: 84.25925925925925, Test acc of LeakyReLu: 87.22222222222223

