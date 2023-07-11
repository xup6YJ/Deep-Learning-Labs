
import numpy as np
import matplotlib.pyplot as plt

'''
1. Implement simple neural networks with two hidden layers.
2. Each hidden layer needs to contain at least one transformation (CNN,
Linear … ) and one activate function ( Sigmoid, tanh….).
3. You must use backpropagation in this neural network and can only use
Numpy and other python standard libraries to implement.
4. Plot your comparison figure that shows the predicted results and the
ground-truth.
5. Print the training loss and testing result as the figure listed below.
'''

np.random.seed(1)

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

# -----------------------
#   activation function & loss
# -----------------------

# Sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
# derivative sigmoid
def derivativeSigmoid(y):
    return np.multiply(y, 1.0 - y)

# Tanh function
def tanh(x):
    return np.tanh(x)
# Derivative of Tanh
def derivativeTanh(y):
    return 1 - y ** 2

# ReLU function
def ReLU(x):
    return np.maximum(0.0, x)

# derivative ReLU
def derivativeReLU(y):
    return np.heaviside(y, 0.0)

# Leaky ReLU function
def LReLU(x):
    return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)

# derivative ReLU
def derivativeLReLU(y):
    y[y > 0.0] = 1.0
    y[y <= 0.0] = 0.01
    return y

def derivative_mse(y, pred_y):
    return -2 * (y - pred_y) / y.shape[0]

def loss_mse(y, pred_y):
    return np.mean((y - pred_y)**2)


class network_h2():
    def __init__(self,  hid_num = 2, neuro_num = 10, activation = 'relu'):

        self.activation = activation

        self.hid_num = hid_num
        self.neuro_num = neuro_num
        # self.layer_num = hid_num + 1
        # self.units = []
        # self.units.append(2)
        # for _ in range(hid_num):
        #     self.units.append(self.neuro_num)
        # self.units.append(1)
        # print('Neurons: ', self.units)

        # self.momentum = np.zeros((in_channel + 1, out_channel))

        self.wsum_0 = 0
        self.wsum_1 = 0
        self.wsum_2 = 0

        self.w0 = np.random.normal(0, 1, (2, self.neuro_num))
        self.w1 = np.random.normal(0, 1, (self.neuro_num , self.neuro_num))
        self.w2 = np.random.normal(0, 1, (self.neuro_num, 1))

        self.momentum0 = np.zeros((2, self.neuro_num))
        self.momentum1 = np.zeros((self.neuro_num , self.neuro_num))
        self.momentum2 = np.zeros((self.neuro_num, 1))

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.w0

        if self.activation =='sigmoid':
            self.a1 = sigmoid(self.z1)
            self.z2 = self.a1 @ self.w1
            self.a2 = sigmoid(self.z2)
            self.z3 = self.a2 @ self.w2

        elif self.activation == 'relu':
            self.a1 = ReLU(self.z1)
            self.z2 = self.a1 @ self.w1
            self.a2 = ReLU(self.z2)
            self.z3 = self.a2 @ self.w2
        
        elif self.activation == 'tanh':
            self.a1 = tanh(self.z1)
            self.z2 = self.a1 @ self.w1
            self.a2 = tanh(self.z2)
            self.z3 = self.a2 @ self.w2
        
        else: #without activation
            self.z2 = self.z1 @ self.w1
            self.z3 = self.z2 @ self.w2

        if self.activation is not None:
            self.yhat = sigmoid(self.z3)
        else:
            self.yhat = self.z3
            
        return self.yhat


    def backward(self, y):
        '''
        backward propagation
        '''
        #output

        self.de_dy = derivative_mse(y, self.yhat) #loss

        if self.activation is not None:
            #W2
            self.d2_act = derivativeSigmoid(self.yhat)  #act the output of sigmoid
            self.bw2 = np.multiply(self.d2_act, self.de_dy) #loss*act (G1)
            # self.backwardGrad_w2 = np.matmul(self.backwardGrad, self.w_dic['w2'].T)  #bw2 = (loss*act)@w2 
            self.fw2 = self.a2.T @ self.bw2 #(10, 1)

        if self.activation =='sigmoid':
            #w1
            self.d1_act = derivativeSigmoid(self.a2) #act() the output of sigmoid
            self.bw1 = np.multiply(self.d1_act, np.matmul(self.bw2, self.w2.T))  #act(z2) * bw2  (G2)
            self.fw1 = self.a1.T @ self.bw1 #(10, 10)
            
            #w0
            self.d0_act = derivativeSigmoid(self.a1) #act() the output of sigmoid #(100, 10)
            self.bw0 = np.multiply(self.d0_act, np.matmul(self.bw1, self.w1.T))  #act(z1) * bw1  (G3)
            self.fw0 = self.x.T @ self.bw0  #shape shoulb be (2, 10)  

        elif self.activation == 'relu':
            #w1
            self.d1_act = derivativeReLU(self.a2) #act() the output of sigmoid
            self.bw1 = np.multiply(self.d1_act, np.matmul(self.bw2, self.w2.T))  #act(z2) * bw2  (G2)
            self.fw1 = self.a1.T @ self.bw1 #(10, 10)
            
            #w0
            self.d0_act = derivativeReLU(self.a1) #act() the output of sigmoid #(100, 10)
            self.bw0 = np.multiply(self.d0_act, np.matmul(self.bw1, self.w1.T))  #act(z1) * bw1  (G3)
            self.fw0 = self.x.T @ self.bw0  #shape shoulb be (2, 10)  

        elif self.activation == 'tanh':
            #w1
            self.d1_act = derivativeTanh(self.a2) #act() the output of sigmoid
            self.bw1 = np.multiply(self.d1_act, np.matmul(self.bw2, self.w2.T))  #act(z2) * bw2  (G2)
            self.fw1 = self.a1.T @ self.bw1 #(10, 10)
            
            #w0
            self.d0_act = derivativeTanh(self.a1) #act() the output of sigmoid #(100, 10)
            self.bw0 = np.multiply(self.d0_act, np.matmul(self.bw1, self.w1.T))  #act(z1) * bw1  (G3)
            self.fw0 = self.x.T @ self.bw0  #shape shoulb be (2, 10)  
        
        else: #w/o activation
            #w2
            dy_dw2 = self.z2
            self.fw2 = dy_dw2.T @ self.de_dy #(10, 1)
            #w1
            da2_dw2 = self.z1
            self.bw1 = np.matmul(self.de_dy, self.w2.T)
            self.fw1 = da2_dw2.T @ self.bw1 #(10, 10)
            #w0
            da1_dw1 = self.x
            self.bw0 = np.matmul(self.bw1, self.w1.T)
            self.fw0 = da1_dw1.T @ self.bw0  #shape shoulb be (2, 10)  


    def update(self, lr, optimizer=None):
        '''
        Update weight
        '''
        self.lr = lr
        # print('optimizer = ', optimizer)

        if optimizer == 'adagrad':
            self.wsum_0 += (self.w0**2)
            self.wsum_1 += (self.w1**2)
            self.wsum_2 += (self.w2**2)

            self.w0 +=  -self.lr * self.fw0 / (self.wsum_0**0.5)
            self.w1 +=  -self.lr * self.fw1 / (self.wsum_1**0.5)
            self.w2 +=  -self.lr * self.fw2 / (self.wsum_2**0.5)

        elif optimizer == 'sgd':
            self.w0 +=  -self.lr * self.fw0
            self.w1 +=  -self.lr * self.fw1
            self.w2 +=  -self.lr * self.fw2

        elif optimizer == 'momentum':
            self.momentum0 = 0.9 * self.momentum0 - self.lr * self.fw0
            self.momentum1 = 0.9 * self.momentum1 - self.lr * self.fw1
            self.momentum2 = 0.9 * self.momentum2 - self.lr * self.fw2

            self.w0 +=  self.momentum0
            self.w1 +=  self.momentum1
            self.w2 +=  self.momentum2
            



def train(model,  epoch = 1000, lr = 1e-3, optimizer = 'sgd', dataType = 'Linear'):
    loss_list = []
    epoch_list = []

    if dataType == 'Linear':
        x, y = generate_linear()
    else:
        x,y = generate_XOR_easy()
    
    model = model
    activation = model.activation

    for epoch in range(epoch):
        pred_y = model.forward(x)
        loss = loss_mse(y, pred_y)
        dL = derivative_mse(y, pred_y)
        model.backward(y)
        model.update(lr, optimizer = optimizer)

        if (epoch+1) % 1000 == 0:
            print(f'epoch {epoch+1} loss : {loss}')
        loss_list.append(loss)
        epoch_list.append(epoch+1)

    pred_y = model.forward(x)
    print('prediction:', pred_y)

    print(f'\nDatatype: {dataType}, Optimizer: {optimizer}, Activation fun: {activation}')
    print('Test loss : ', loss_mse(pred_y, y))
    show_result(x, y, pred_y)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0

    print(f'Test Acc: {np.sum(pred_y == y) / y.shape[0] * 100}%')
   
    #plot
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'{dataType}_{optimizer}_{activation}_{model.hid_num}_{model.neuro_num}.png')
    plt.show()
    plt.close()

if __name__ == '__main__':

 
    model = network_h2(activation = 'relu')

    '''
    optimizer
    '''
    #100%
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'XOR')
    train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'Linear')

    #100%
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'adagrad', dataType = 'XOR')
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'adagrad', dataType = 'Linear')

    #100%
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'momentum', dataType = 'XOR')
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'momentum', dataType = 'Linear')

    '''
     dif activation with sgd
     '''
    # 76.2%/ 100%
    # model = network_h2(hid_num = 2, neuro_num = 2, activation = 'sigmoid')
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'momentum', dataType = 'XOR')
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'linear')

    #71.2% /100% 
    # model = network_h2(hid_num = 2, neuro_num = 2, activation = 'tanh')
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'momentum', dataType = 'XOR')
    # train(model = model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'linear')

    #33% / 90%
    # model = network_h2(hid_num = 2, neuro_num = 2, activation = None)
    # train(model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'XOR')
    # train(model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'linear')

    '''
     dif neurons
     '''
    #46%, 76%
    # model = network_h2(hid_num = 2, neuro_num = 2, activation = 'relu')
    # train(model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'linear')
    # model = network_h2(hid_num = 2, neuro_num = 2, activation = 'relu')
    # train(model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'XOR')

    #100%
    # model = network_h2(hid_num = 2, neuro_num = 10, activation = 'relu')
    # train(model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'linear')
    # model = network_h2(hid_num = 2, neuro_num = 10, activation = 'relu')
    # train(model, epoch = 15000, lr = 1e-1, optimizer = 'sgd', dataType = 'XOR')




