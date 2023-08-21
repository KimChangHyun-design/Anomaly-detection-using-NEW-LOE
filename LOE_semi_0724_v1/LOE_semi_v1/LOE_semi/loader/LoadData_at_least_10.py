# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from scipy import io
from .utils import *
import torch
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import *

def CIFAR10_feat(normal_class,root='DATA/cifar10_features/',contamination_rate=0.0):
    trainset = torch.load(root+'trainset_2048.pt')
    train_data,train_targets = trainset
    testset = torch.load(root+'testset_2048.pt')
    test_data,test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets==normal_class]=0
    
    train_clean = train_data[train_targets==normal_class]
    train_contamination = train_data[train_targets!=normal_class]
    num_clean = train_clean.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]),num_contamination,replace=False)
    train_contamination = train_contamination[idx_contamination]    
    train_data = torch.cat((train_clean,train_contamination),0)
        
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1

    return train_data,train_labels,test_data,test_labels


def FMNIST_feat(normal_class, root='DATA/fmnist_features/',contamination_rate=0.0):
    trainset = torch.load(root + 'trainset_2048.pt')
    train_data, train_targets = trainset
    testset = torch.load(root + 'testset_2048.pt')
    test_data, test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets == normal_class] = 0

    train_clean = train_data[train_targets == normal_class]
    train_contamination = train_data[train_targets != normal_class]
    num_clean = train_clean.shape[0]
    num_contamination = int(contamination_rate / (1 - contamination_rate) * num_clean)

    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]), num_contamination, replace=False)
    train_contamination = train_contamination[idx_contamination]
    train_data = torch.cat((train_clean, train_contamination), 0)

    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:] = 1

    return train_data, train_labels, test_data, test_labels

def synthetic_contamination(norm,anorm,contamination_rate):
    num_clean = norm.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    try:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=False)
    except:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=True)
    train_contamination = anorm[idx_contamination]
    train_contamination = train_contamination  + 0.03 * np.random.randn(*train_contamination.shape)
    train_data = np.concatenate([norm,train_contamination],0)
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1
    return train_data,train_labels

def synthetic_contamination_v2(norm,anorm,contamination_rate, true_anomaly_rate):
    num_clean = norm.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    try:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=False)
    except:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=True)
    train_contamination = anorm[idx_contamination]
    train_contamination = train_contamination  + 0.03 * np.random.randn(*train_contamination.shape)
    train_data = np.concatenate([norm,train_contamination],0)
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1
    train_true_labels =  np.zeros(train_data.shape[0])
    train_true_labels[-int(num_contamination*(true_anomaly_rate)):]=1
    return train_data,train_labels,train_true_labels


def Thyroid_data(contamination_rate):
    data = io.loadmat("DATA/thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    inliers = samples[labels == 0]  # 3679 norm
    outliers = samples[labels == 1]  # 93 anom

    num_split = len(inliers) // 2
    train_norm = inliers[:num_split]  # 1839 train
    test_data = np.concatenate([inliers[num_split:], outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1

    train, train_label = synthetic_contamination(train_norm, outliers, contamination_rate)
    return train, train_label, test_data, test_label


def Arrhythmia_data(contamination_rate):
    data = io.loadmat("DATA/arrhythmia.mat")
    samples = data['X']  # 518
    labels = ((data['y']).astype(np.int32)).reshape(-1)#452

    inliers = samples[labels == 0]  # 386 norm
    outliers = samples[labels == 1]  # 66 anom

    num_split = len(inliers) // 2
    #inlier 데이터의 반절을 train_norm으로 정의
    train_norm = inliers[:num_split]  # 193 train
    #inlier 데이터의 남은 반절과  outlier 데이터 셋을 결합
    test_data = np.concatenate([inliers[num_split:], outliers], 0)# 259 
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1
    
    # contaminated 된 갯수 만큼 anomaly를 추가하여 학습
    train, train_label = synthetic_contamination(train_norm, outliers, contamination_rate)
    #cotanmination ratio = 0.1
    #train 214 test 259 ## 193*0.1/0.9=21 
    #Therefore, 193+21 =214
    return train, train_label, test_data, test_label


def csagn(contamination_rate, true_anomaly_rate):
    data_path = 'DATA/csagn/'
    train = np.load(data_path+ 'train_101_array.npy')
    train_label = np.load(data_path + 'train_101_label.npy')
    test = np.load(data_path + 'test_101_array.npy')
    test_label = np.load(data_path + 'test_101_label.npy')
    samples = np.concatenate([train, test], 0)# 1026
    labels = np.concatenate([train_label, test_label], 0) #1026

    inliers = samples[labels == 0]  # 1023
    outliers = samples[labels == 1]  # 3 anom

    num_split = len(inliers) // 2
    #inlier 데이터의 반절을 train_norm으로 정의
    train_norm = inliers[:num_split]  # 511 train
    #inlier 데이터의 남은 반절과  outlier 데이터 셋을 결합
    test_data = np.concatenate([inliers[num_split:], outliers], 0)#515 
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1
    
    # contaminated 된 갯수 만큼 anomaly를 추가하여 학습
    #train, train_label = synthetic_contamination(train_norm, outliers, contamination_rate)
    train, train_label, train_true_label = synthetic_contamination_v2(train_norm, outliers, 0.1, true_anomaly_rate)
    #known anomaly labeled as 2 
    #train_label[495:505]=2
    #cotanmination ratio = 0.1
    #train 495 test 580 ## 495*0.1/0.9=55 
    #Therefore, 495 +55=214
    train = np.transpose(train,(0,2,1))
    test_data = np.transpose(test_data,(0,2,1))
    train=train[:,1:,:]
    test_data=test_data[:,1:,:]
    return train, train_label, test_data, test_label, train_true_label



def synthetic_contamination(norm,anorm,contamination_rate):
    num_clean = norm.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    try:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=False)
    except:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=True)
    train_contamination = anorm[idx_contamination]
    train_contamination = train_contamination  + 0.03 * np.random.randn(*train_contamination.shape)
    train_data = np.concatenate([norm,train_contamination],0)
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1
    return train_data,train_labels


def lg_data_ntl_plus(contamination_rate,true_anomaly_rate):
    # 제품의 csv 파일 입력 부분
    #data = pd.read_csv("DATA/lg/CalibrationSample_M870AAA452.csv",index_col=0) # done
    #data = pd.read_csv("DATA/lg/CalibrationSample_M871GBB551.csv",index_col=0) # done--
    #data = pd.read_csv("DATA/lg/CalibrationSample_M870AAA451.csv",index_col=0) # done 
    #data = pd.read_csv("DATA/lg/CalibrationSample_W822AAA152.csv",index_col=0) # done
    #data = pd.read_csv("DATA/lg/CalibrationSample_M870AAA451.csv",index_col=0)# done
    data = pd.read_csv("DATA/lg/CalibrationSample_M872AAA031.csv",index_col=0)# done-- 
    
    
    data = data.reset_index()
    # 1. len_filter 적용
    # data_len = data[data['len_filter']==1].reset_index(drop=True)
    # 2. RESULT 적용
    #  data_len_ok = data_len[data_len['RESULT']=='OK']
    #  data_len_ok = data_len_ok.reset_index(drop=True) # reset index안해주면 len_barcode_seq가 안됨
    # 최종 사용할 data는 data_len_ok 임

    # LG 측에서 제공한 실제 고장 BARCODE_hash 입력
    #barcode_anomaly = ['gAAAAABhwo7BC3xrJ17V8mkX4-3sJ1S9xu0ltGqP8-ApcNxRD2hG8Hsc6Aa923ERs-sLsM4M2U68fZSkjTW5lTxWnhOiw3IdSwC-TkcbtKVveBNN6sUqL74='] # done
    #barcode_anomaly = ['gAAAAABhwo7CLznfZitvOF1Zt2S9DefyDvFi2hUbuYhtzKM-5yQcj-fsq-SK-s_9lF5VObM7hpSnkwmA20slXnCplQEfoDLX-wl9l0wkHCgZYXgGNr9ItfQ='] #done
    #barcode_anomaly = ['gAAAAABhwo7C4eAPjtOdXWTW9tE84sjGYNtCGTEiK4gIaoSd_yU4ZPVSzL9B1nkHpSk0Mz1-6ONDExbPlqPRy3p6oimujuSGvYFYvv7KFNSAFe4NQDj8wVE='] # done
    #barcode_anomaly = ['gAAAAABib2XPayWAPUSOLHyWifIKhZHHszb4nNckJ86bAMLA3bpJzL-2JkwQrM_lsp-ic-UqC6GG2q1rh3m7ao4HwP2RnGtLaHZI79xRiJOoTee9EGsdEkM='] # done
    #barcode_anomaly = ['gAAAAABhwo7CJ2ay49VLou1_ff08fWfmjTJoU8OFB2yFvK-IWGSaqd5_Eu3i66AVsc3CKPlfKLr0lkJIaZEsj7t3FNDpZ3vq7q-3U8HqUFEwxCy1at-wl7c=']# done
    barcode_anomaly = ['gAAAAABib2XRWkbhm2BsowH5oJK2rmYbsEiSsduDeaFqQlaFBDarSnzai2YoUN70EirGuxEDr0yNwN4gC7yUp8Nu3xLrWyIZ6glqSe_T4-CPaoKHHsnpVtc=']# done

    
    
    # 실제 고장 BARCODE_hash 입력받아 나오는 최종 고장 데이터 추출
    #(2598, 63)
    data_anomaly = data[data['BARCODE_hash'].isin(barcode_anomaly)]
    data_anomaly = data_anomaly.reset_index(drop=True)

    # 실제 고장 데이터 제외 정상 데이터
    #(4184079, 63)
    data_normal = data[~data['BARCODE_hash'].isin(barcode_anomaly)]
    data_normal = data_normal.reset_index(drop=True)

    # 정상 데이터와 실제 고장 데이터를 합쳐주는 과정
    ##(4186677, 63)
    final_samples = pd.concat([data_normal, data_anomaly], axis=0)
    final_samples=final_samples.reset_index(drop=True)

    # 정상 데이터와 실제 고장 데이터를 합친 전체 sample 개수 구함
    #(3223,)
    unique_ID = final_samples['BARCODE_SEQ'].unique()

    # 사용할 변수들 입력하는 과정(multivariate) , 추 후 더 입력이 가능함
    #output_features = ['F-DefrostSensorTemperature', 'R-DefrostSensorTemperature', 'Comp-Power', 'Comp-Phase', 'Comp-Current', 'Comp-Stroke']
    output_features = ['F-DefrostSensorTemperature', 'R-DefrostSensorTemperature', 'Comp-Power']
    #output_features = ['Comp-Power']

    # 딥러닝 모델을 적용하기 위해 final_samples에 스케일링 진행
    #(4186677, 6)
    scaler = StandardScaler()
    train_X_norm=scaler.fit_transform(final_samples.loc[:, output_features])
    # time-series 고려하기 위해 (sample갯수, sequence 길이, 변수 개수)로 변환
    #(3223, 1299, 6)
    x_train = train_X_norm.reshape(len(unique_ID), 1299, train_X_norm.shape[1])

    samples = x_train

    labels = np.zeros(samples.shape[0])
    # 고장 데이터 라벨 붙여주는 과정
    # 2 due to 2598/1299=2
    labels[samples.shape[0]-data_anomaly['BARCODE_SEQ'].unique().shape[0]:samples.shape[0]]=1

    # 최종적으로 전처리가 완료된 sample과 label 

    samples = np.array(samples) 
    labels = np.array(labels)

    inliers = samples[labels == 0]  # 
    outliers = samples[labels == 1]  # 1 for anomalies

    num_split = len(inliers) // 2
    train_norm = inliers[:num_split]  # 1610,1299,6

    #inlier 데이터의 남은 반절과  outlier 데이터 셋을 결합
    test_data = np.concatenate([inliers[num_split:], outliers], 0)#1611+2
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1

    #train, train_label,train_true_label = synthetic_contamination_v2(train_norm, outliers, contamination_rate,true_anomaly_rate)
    train, train_label,train_true_label = synthetic_contamination_v2(train_norm, outliers, 0.1,true_anomaly_rate)

    train = np.transpose(train,(0,2,1))
    test_data = np.transpose(test_data,(0,2,1))

    return train, train_label, test_data, test_label,train_true_label


def simulation_data_ntl_plus(simu_,contamination_rate,true_anomaly_rate):
    simu_=str(simu_)
    contamination = str(contamination_rate)
    train = pd.read_csv("DATA/simulation/train/simulation"+simu_+"_"+contamination+"_train.csv")
    test = pd.read_csv("DATA/simulation/test/simulation"+simu_+"_"+contamination+"_test.csv")

    train_label = np.array(train['label'])
    train = np.array(train.iloc[:,:100]) #2000,100 2000 samples*100 observatrion

    test_label = np.array(test['label']) 
    test = np.array(test.iloc[:,:100]) #2000,100 2000 samples*100 observatrion

    num_clean = len(train_label[train_label==0]) #1800
    num_anorm = len(train_label[train_label==1]) #200

    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean) # 200
    num_true_anomaly=int(num_contamination*true_anomaly_rate)#200*0.1 =20 

    #idx_contamination = np.random.choice(np.arange(num_anorm),num_true_anomaly,replace=False)
    idx_true = np.random.choice((train_label > 0).nonzero()[0],num_true_anomaly,replace=False)
    train_true_label=np.zeros(train_label.shape[0])
    train_true_label[idx_true]=1

    train=train.reshape(train.shape[0], 1, train.shape[1])
    test=test.reshape(test.shape[0], 1, test.shape[1])
    
    return train, train_label, test, test_label, train_true_label




# def simulation_data_ntl_plus(simu_,contamination_rate,true_anomaly_rate):
#     simu_=str(simu_)
#     contamination = str(contamination_rate)
#     train = pd.read_csv("DATA/simulation/train/simulation"+simu_+"_"+contamination+"_train.csv")
#     test = pd.read_csv("DATA/simulation/test/simulation"+simu_+"_"+contamination+"_test.csv")
    
#     train_label = np.array(train['label'])
#     train = np.array(train.iloc[:,:100]) #2000,100 2000 samples*100 observatrion
    
#     test_label = np.array(test['label']) 
#     test = np.array(test.iloc[:,:100]) #2000,100 2000 samples*100 observatrion
#     samples = np.concatenate([train, test], 0)#3000
#     labels = np.concatenate([train_label, test_label], 0)

#     inliers = samples[labels == 0]# 2700
#     outliers = samples[labels == 1]  # 300 anom
#     num_split = len(inliers) // 2 #1350
#     #inlier 데이터의 반절을 train_norm으로 정의
#     train_norm = inliers[:num_split]  # 1350 train

#     #inlier 데이터의 남은 반절과  outlier 데이터 셋을 결합
#     test_data = np.concatenate([inliers[num_split:], outliers], 0)#1350+300=1650 
#     test_label = np.zeros(test_data.shape[0])
#     test_label[num_split:] = 1

#     # contaminated 된 갯수 만큼 anomaly를 추가하여 학습
#     #cotanmination ratio = 0.1
#     #train 1350 test 1650 ## 1350*0.1/0.9=150
#     #Therefore, 1350 +150=1500
#     #if contaminated ratio 0.1
#     #print(list(train_true_label).count(1)) 
#     ##15 
#     train, train_label, train_true_label = synthetic_contamination_v2(train_norm, outliers, contamination_rate, true_anomaly_rate)
#     train = train.reshape(train.shape[0], 1, train.shape[1])
#     test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
#     return train, train_label, test_data, test_label, train_true_label



def load_data(simu_,data_name,cls,contamination_rate=0.0, true_anomaly_rate=0.0):

    ## normal data with label 0, anomalies with label 1

    if data_name == 'cifar10_feat':
        train, train_label, test, test_label = CIFAR10_feat(cls,contamination_rate=contamination_rate)
    elif data_name == 'fmnist_feat':
        train, train_label, test, test_label = FMNIST_feat(cls, contamination_rate=contamination_rate)
    elif data_name == 'thyroid':
        train, train_label, test, test_label = Thyroid_data(contamination_rate)
    elif data_name == 'arrhythmia':
        train, train_label, test, test_label = Arrhythmia_data(contamination_rate)
    elif data_name == 'comp_data':
        train, train_label, test, test_label = comp_data(contamination_rate)
    elif data_name == 'simulation_data_ntl_plus' :
        train, train_label, test, test_label,true_train_label = simulation_data_ntl_plus(simu_,contamination_rate,true_anomaly_rate)   
    elif data_name == 'lg_data_ntl_plus' :
        train, train_label, test, test_label,true_train_label = lg_data_ntl_plus(contamination_rate,true_anomaly_rate)
    elif data_name == 'csagn' :
        train, train_label, test, test_label, true_train_label = csagn(contamination_rate, true_anomaly_rate)   
    trainset = CustomDataset(train,train_label, true_train_label)
    testset = CustomDataset(test,test_label, test_label )
    return trainset,testset,testset






