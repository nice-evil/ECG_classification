#====================================================== 导入需要的包==================================
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
import os
import platform
import datetime
import seaborn
import tensorflow as tf
import keras.backend as K

from keras.utils.np_utils import *
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn import preprocessing
# 调用自己写的模型架构
from model.CNN import CNN1d
from model.CNNLSTM import CNN_LSTM
from model.SENetLSTM import SENet_LSTM
from model.GRU_CNN import Seq_SE_GRU
#=================================================== 读取数据================================================
np.random.seed(7)
choose_index=np.random.randint(1,100,100)
print("load data...")

X=np.loadtxt('./waverec_Mit_arr_X_eu_MLII.csv',delimiter=',',skiprows=1).astype('float32')#[choose_index]
Y=np.loadtxt('./waverec_Mit_arr_change_Y_eu_MLII.csv',dtype="str",delimiter=',',skiprows=1)#[choose_index]

#几种目标需求的分类label
AAMI=['V','j','L','J','R','E','B']
# N:Normal
# L:Left bundle branch block beat
# R:Right bundle branch block beat
# V:Premature ventricular contraction
# A:Atrial premature contraction
# |:Isolated QRS-like artifact
# B:Left or right bundle branch block
delete_list=[]
for i in range(len(Y)):
    print(i)
    if Y[i] not in AAMI:            # 删除不在AAMI中标签的数据
        delete_list.append(i)
X=np.delete(X,delete_list,0)#0:按行删除
Y=np.delete(Y,delete_list,0)

#保存用于训练的数据
savedX=pd.DataFrame(X)
savedY=pd.DataFrame(Y)
#统计用于训练的各类的个数pd.read_csv()
column_name=savedY.columns[0]
column_data=savedY[column_name]
print("心电类型统计: ", dict(column_data.value_counts()))
savedX.to_csv('./data/Train_X.csv', index=False)
savedY.to_csv('./data/Train_Y.csv', index=False)

#数据标准化x-u/q：
print("begin standard scaler...")
ss = StandardScaler()
std_data = ss.fit_transform(X)
print(std_data)
X=np.expand_dims(X,axis=2)

# 把标签编码,将mit标注格式转化为range的数字，方便训练
le=preprocessing.LabelEncoder()
le=le.fit(AAMI)
Y=le.transform(Y)
print("the label before encoding:",le.inverse_transform([0,1,2,3,4,5,6]))

# 定义超参数
num_epochs=20
batch_size = 32
learning_rate = 0.001
# 分层抽样
print("begin StratifiedShuffleSplit...")
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, train_size=0.7,random_state=0)

#===================================================模型训练==================================================
input = tf.keras.layers.Input(shape=(360, 1))
output=SENet_LSTM(input)
print("begin CNNLSTM")
model = Model(input, output)
print('model summary:', model.summary())
# 设置优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, name='Adam')
loss = tf.keras.losses.categorical_crossentropy
metrics = ['accuracy']
# 初始化tensorboard
# 注意windows系统和linux系统下文件路径的问题
if platform.system() == 'Windows':
    log_dir = "tensorboard\\fit\\CNN" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
elif platform.system() == 'Linux':
    log_dir = "tensorboard/fit/CNN" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)  # 实例化一个记录器
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# 配置训练过程
print('begin compile CNN1d')
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#储存5折交叉验证的每次信息
test_acc_all=[]
num=0
for train_index, test_index in sss.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    y_train=to_categorical(y_train,7)

    # 训练模型
    print("begin fit CNN")
    # 使用checkpoint保存最好的模型参数
    print("begin saving checkpoint...")
    filepath="weights.best.hdf5"
    #只保存val_acc验证集中表现最好的model
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
    callbacks_list = [tensorboard_callback,checkpoint]

    # Fit the model
    saveMoudle=model.fit(X_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_split=0.33,callbacks=callbacks_list)
    # 评估训练效果
    y_test_trans=to_categorical(y_test,7)
    test_loss,test_accuracy=model.evaluate(X_test,y_test_trans)
    print('test_loss:',test_loss)
    print('test_acc:',test_accuracy)
    test_acc_all.append(test_accuracy)
    predict=model.predict(X_test)       # 输出的不是一个类别，而是样本属于每一个类别的概率
    predict=[np.argmax(predict[i]) for i in range(len(predict))]

    #画出混淆矩阵
    confusion_matrix=tf.math.confusion_matrix(y_test,predict)
    print('confusion matrix:',confusion_matrix)
    plt.figure()
    seaborn.heatmap(confusion_matrix, annot=True, fmt='.20g', cmap='Blues')

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    tick_marks = np.arange(len(AAMI))
    AAMI = ['V', 'j', 'L', 'J', 'R', 'E', 'B']
    #通过改变xticks讲刻度显示放在中间
    plt.xticks([index + 0.5 for index in tick_marks], AAMI, rotation='horizontal')
    plt.yticks([index + 0.5 for index in tick_marks], AAMI, rotation='horizontal')
    plt.title('Confusion Matrix')
    plt.savefig('SENet_LSTM_confusion_matrix.png')
    #图片显示3s后自动关闭
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

print('test_acc_all:',test_acc_all)
