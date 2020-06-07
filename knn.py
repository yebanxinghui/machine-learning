#k近邻算法实现
import numpy as np
import operator
import os
import gzip

train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'

#按32位读取，主要为读校验码、图片数量、尺寸准备的
def read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#抽取图片
def extract_images(input_file,value_binary):
    with gzip.open(input_file, 'rb') as zipf:
        magic = read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = read32(zipf)
        rows = read32(zipf)
        cols = read32(zipf)
        print(magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if value_binary:
            data = data.reshape(num_images, rows * cols)
        else:
            data = data.reshape(num_images, rows , cols)
        return np.minimum(data, 1)

#抽取标签
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

train_x = extract_images(train_images,True)
train_y = extract_labels(train_labels)
test_x = extract_images(test_images,True)
test_y = extract_labels(test_labels)

#使用散点图分析
import matplotlib
import matplotlib.pyplot as plt
def print_image(image):
    plt.imshow(image,cmap = 'gray')
    plt.pause(0.0001)
    plt.show()
    
def plotk(k):
    test_num = test_x.shape[0] // 10
    matchCount = 0
    for i in range(test_num):
        print('该图片周围的图片有：')
        predict = classify0(test_x[i], train_x, train_y, k)
        if predict == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / test_num
    return accuracy


#kNN算法具体实现,其中有四个输入参数
#用于分类的输入向量inX，输入的训练样本集为dataSet,标签向量为labels，用于选择最近邻的数目为k
#其中的距离度量为欧几里得距离
#来源于机器学习实战书籍第一章
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #数据集的行数
    init_shape = inX.shape[0]
    inX = inX.reshape(1, init_shape)
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = np.sum(sqDiffMat,axis=1)
    #对于二维数组,axis=1表示按行相加 , axis=0表示按列相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = np.argsort(distances)
    #X.argsort()将X中的元素从小到大排序，提取其对应的index（索引）并输出,但X本身的次序没变动
    classCount={}
    train_x = extract_images(train_image,False)
    train_x = train_x.reshape(dataSetSize,28,28)
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        print_image(train_x[sortedDistIndicies[i]])#
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    #operator.itemgetter(1)为定义了一个取第一个域值的函数
    return sortedClassCount[0][0]


fig = plt.figure()

K = range(1,10)
listk = []
for k in K:
    listk.append(1-plotk(k))
plt.plot(K,listk,'bx-')
plt.xlabel('k')
plt.ylabel('misclassification rate')
plt.title('k : misclassification rate')
plt.show()

#[0.964, 0.964, 0.962, 0.963, 0.958, 0.959, 0.956, 0.956, 0.952]
#[0.964, 0.953, 0.954, 0.947, 0.941, 0.939, 0.934, 0.93, 0.927, 0.926, 0.926]