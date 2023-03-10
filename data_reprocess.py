import wfdb     #导入wfdb包读取数据文件
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
from scipy import signal
import pywt

#提取所有文件中导联中含有“MLII”导联的数据并进行类别统计
def extract_person_data(rootdir):
    type = []
    files = os.listdir(rootdir) #列出文件夹下所有
    name_list=[]            # name_list=[100,101,...234]
    MLII=[]                 # 用MLII型导联采集的人（根据选择的不同导联方式会有变换）
    type={}                 # 标记及其数量
    for file in files:
        if file[0:3] in name_list:     # 选取文件的前3个字符（可以根据数据文件的命名特征进行修改）
            continue
        else:
            name_list.append(file[0:3])
    for name in name_list:      # 遍历每一个人
        if name[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:  # 判断——跳过无用的文件
            continue
        record = wfdb.rdrecord(rootdir+'/'+name)  # 读取一条记录（100），不用加扩展名
        filenames=record.sig_name
        if 'MLII' in record.sig_name:       # 这里我们记录MLII导联的数据（也可以记录其他的，根据数据库的不同选择数据量多的一类导联方式即可）
            MLII.append(name)               # 记录下这个人
        annotation = wfdb.rdann(rootdir+'/'+name, 'atr')  # 读取一条记录的atr文件，扩展名atr
        for symbol in annotation.symbol:            # 记录下这个人所有的标记类型
            if symbol in list(type.keys()):
                type[symbol]+=1
            else:
                type[symbol]=1
        print('symbol_name',type)
    sorted(type.items(),key=lambda d:d[1],reverse=True)
    return MLII

# 小波变换降噪
# wavelet denoise preprocess using mallat algorithm
def denoise(data):
    # 小波分解
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 软阈值降噪、提取主要心电成分
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 重构信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

#进行数据切割
def extract_epoch_data(MLII):
    f=360       # 原始采样频率
    segmented_len=1 #裁剪时间为1s,anotation前后0.5s
    label_count=0
    abnormal=0

    segmented_data = []             # 最后数据集中的X
    segmented_label = []            # 最后数据集中的Y
    print('begin!')

    for person in MLII:        # 读取导联方式为MLII的数据
        whole_signal=wfdb.rdrecord(rootdir + '/' + person).p_signal.transpose() # 读取某人的一整条记录
        whole_annotation = wfdb.rdann(rootdir + '/' + person, 'atr')  # 读取一条记录的atr文件，扩展名atr
        #标记位置
        Rlocation = whole_annotation.sample
        for i in range(len(whole_annotation.sample)):
            #使用MIT-BIH数据集提供的人工标注，并在尖峰处向前取0.5s、向后取0.5s，以提取一个完整的心拍
            #每段提取的起点，起始点必须为大于0的点
            sampfrom = int(max(whole_annotation.sample[i] - (f * segmented_len) / 2, 0))
            #每段提取的终止点，终点不能大于最大数据长度
            sampto = int(min(whole_annotation.sample[i] + (f * segmented_len) / 2, len(whole_signal[0])))
            record=wfdb.rdrecord(rootdir + '/' + person, sampfrom=sampfrom,sampto=sampto)
            annotation = wfdb.rdann(rootdir + '/' + person, 'atr',sampfrom=sampfrom,sampto=sampto)
            symbols = annotation.symbol
            if symbols.count('N') / len(symbols) == 1:  # 如果全是'N',则跳过这条正常心拍，只记录异常心拍
                continue
            else:
                lead_index = record.sig_name.index('MLII')  # 找到MLII导联对应的索引
                signal = record.p_signal.transpose()  # 两个导联，转置之后方便画图 ndarray(2,300)
                re_signal = scipy.signal.resample(signal[lead_index], 360) #(360,)
                # plt.plot(re_signal)
                # plt.show()
                #小波降噪
                re_signal_3= denoise(re_signal) #(360,)
                # plt.plot(re_signal_3)
                # plt.show()
                segmented_data.append(re_signal_3)
                print('symbols', symbols, len(symbols))

                label=[]
                # if '+' in symbols:  # 删去+
                #     symbols.remove('+')
                if len(symbols) == 0:
                    segmented_label.append('Q')
                elif symbols.count('N') / len(symbols) == 1 or symbols.count('N') + symbols.count('/') == len(symbols):  # 如果全是'N'或'/'和'N'的组合，就标记为N
                      segmented_label.append('N')
                elif symbols.count('N') / len(symbols) == 1:  # 如果全是'N'，就标记为N
                      segmented_label.append('N')
                      continue
                else:
                    for i in symbols:
                        if i != 'N':
                            label.append(i)
                    segmented_label.append(label[0])
    print('begin to save dataset!')
    return segmented_data,segmented_label

if __name__=='__main__':
    rootdir = './data/'
    MLII=extract_person_data(rootdir)
    #print(MLII)
    segmented_data,segmented_label=extract_epoch_data(MLII)
    segmented_data=pd.DataFrame(segmented_data)
    segmented_label=pd.DataFrame(segmented_label)
    segmented_data.to_csv('waverec_Mit_arr_X_eu_MLII.csv', index=False)
    segmented_label.to_csv('waverec_Mit_arr_change_Y_eu_MLII.csv', index=False)

    print('Finished!')
