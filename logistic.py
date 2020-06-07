import numpy as np
import pandas as pd
#对sigmoid函数的优化，避免了出现极大的数据溢出
def sigmoid(inx):
    if inx >= 0:
        return 1.0/(1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))

def train(x_train, y_train, round):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    list_bias = []          #用来存放每3轮训练后的偏差值
    list_weights = []       #用来存放每3轮训练后的权重
    train_acc = []          #用来存放每3轮在训练集上的精度
    train_loss = []          #用来存放每3轮在训练集上的loss
    bias = 0                #偏置值初始化
    weights = np.ones(dim)  #权重初始化
    learning_rate = 1       #初始学习率
    reg_rate = 0.001        #正则项系数
    bg2_sum = 0             #用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim) #用于存放权重的梯度平方和

    for i in range(round+1):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= num
        w_g /= num

        # adagrad
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        
        # 更新权重和偏置
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g
        if i % 3 == 0 and i != 0:
            list_weights.append(weights)
            list_bias.append(bias)
            acc = 0
            loss = 0
            result = np.zeros(num)
            for j in range(num):
                y_pre = weights.dot(x_train[j, :]) + bias
                sig = sigmoid(y_pre)
                if sig >= 0.5:
                    result[j] = 1
                else:
                    result[j] = 0

                if result[j] == y_train[j]:
                    acc += 1.0
                loss += (-1) * (y_train[j] * np.log(sig + 1e-5) + (1 - y_train[j]) * np.log(1 - sig + 1e-5))
            train_acc.append(acc / num)
            train_loss.append(loss / num)
    return list_weights, list_bias, train_acc, train_loss


# 验证模型效果
def validate(x_label, y_label, weights, bias):
    num = 1000
    test_acc = []
    test_loss = []
    length = len(weights)
    for k in range(length):
        loss = 0
        acc = 0
        result = np.zeros(num)
        for j in range(num):
            y_pre = weights[k].dot(x_label[j, :]) + bias[k]
            sig = sigmoid(y_pre)
            if sig >= 0.5:
                result[j] = 1
            else:
                result[j] = 0

            if result[j] == y_label[j]:
                acc += 1.0
            loss += (-1) * (y_label[j] * np.log(sig + 1e-5) + (1 - y_label[j]) * np.log(1 - sig + 1e-5))
        test_acc.append(1 - acc / num)
        test_loss.append(loss / num)
    return test_acc,test_loss

df = pd.read_csv('income.csv',engine='python',header=None)
df = df.fillna(0)
array = np.array(df)
x = array[:, 1:-1]
x[:, -1] /= np.mean(x[:, -1])
x[:, -2] /= np.mean(x[:, -2])
y = array[:, -1]

# 划分训练集与验证集
x_train, x_label = x[0:3000, :], x[3000:4000, :]
y_train, y_label = y[0:3000], y[3000:4000]

#使用散点图分析
import matplotlib.pyplot as plt
plt.figure()
round = 100
w, b, train_acc, train_loss = train(x_train, y_train, round)
test_acc,test_loss = validate(x_label, y_label, w, b)
k = list(range(0,round+1,3))
k = k[1:]
plt.xticks(k)
plt.plot(k,train_acc,'bx-')
plt.xlabel('round')
plt.ylabel('accuracy')
plt.title('round : train accuracy')
plt.show()

plt.xticks(k)
plt.plot(k,train_loss,'bx-')
plt.xlabel('round')
plt.ylabel('loss')
plt.title('round : train loss')
plt.show()

plt.xticks(k)
plt.plot(k,test_loss,'bx-')
plt.xlabel('round')
plt.ylabel('loss')
plt.title('round : test loss')
plt.show()

plt.xticks(k)
plt.plot(k,test_acc,'bx-')
plt.xlabel('round')
plt.ylabel('misclassification rate')
plt.title('round : misclassification rate')
plt.show()
