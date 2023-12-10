from keras.models import Sequential, Model, Input
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv3D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D,ConvGRU2D
#from keras.layers.convolutional_recurrent import AttenIConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import TimeDistributed,Flatten,Dropout,Dense, Activation
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D,MaxPooling3D
import numpy as np
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
from keras.models import Sequential, load_model
#from keras.layers.normalization import LayerNormalization
import tensorflow as tf
import random
import math
from os import listdir
from os.path import isfile, join, isdir
from PIL import Image
import cv2 as cv
from sklearn.cluster import KMeans
import numpy
import collections
from sklearn import metrics
import pandas
from sklearn import cluster
import scipy.stats
from scipy import stats
import timeit
from keras import backend as k
from pylab import mpl
import random
random.seed(300)

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)



class Config:
  DATASET_PATH = "./train22"
  VAL_PATH = "./traintest"
  SINGLE_TEST_PATH = "./test22/87922"
  BATCH_SIZE = 10
  EPOCHS = 1000
  MODEL_PATH = "./model/model_HSV_0.5_10_2000_mse_70_1.hdf5"#"./model_11/model_HSV_0.1_V1.hdf5"
  STATE = False
  SIZE1 = 64
  SIZE2 = 64
  SZ = 738
  DA = 0.5
  CHANNEL = 1
  SEQENCE = 10
  INTERVAL = 2
  START1 = 100
  START2 = 100
  SN = 10
  TEST_NUM = 5
  RELI = False
  CLUS = 2 #簇类数目
  KNUM = 1 #选择相应的簇（中心数值由小到大排序）

def findSmallKnum(arr, k):
    arr1 = arr.reshape(-1,1)
    arr1 = arr1.flatten()
    arr1 = np.sort(arr1)
    return (np.where(arr1==arr[k-1])[0])[0]

def get_clips_by_stride(stride, frames_list, sequence_size):
    clips = []
    sz = len(frames_list)

    clip = np.zeros(shape=(sequence_size, Config.SIZE1, Config.SIZE2, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips

def flat_values(sig, tv):
    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]):
            if sig[i,j] > tv:
                sig[i,j] = sig[i,j]
            else:
                sig[i,j] = 150 - sig[i,j]
    return sig

def getdata(path,n1=1,n2=3,sequence_size=Config.SEQENCE):
    clips = []
    for f in sorted(listdir(path)):
        print(f)
        directory_path = join(path, f)
        if isdir(directory_path):
            all_frames = []
            for c in sorted(listdir(directory_path)):
                img_path = join(directory_path, c)
                print(img_path)
                if str(img_path)[-3:] == "jpg":
                    image = cv.imread(img_path)
                    img = cv.cvtColor(image,cv.COLOR_BGR2HSV)
                    img = cv.resize(img,(Config.SIZE1, Config.SIZE2))
                    img = np.array(img[:,:,2], dtype=np.uint8)
                    img[img<70] = 0
                    img = img / 255
                    img =  img.reshape(-1,1)
                    all_frames.append(img.reshape(Config.SIZE1,Config.SIZE2))
            for stride in range(n1,n2):
                clips.extend(get_clips_by_stride(stride=stride,frames_list=all_frames,sequence_size=sequence_size))
    return clips

a = Config.DA

def mse_ca(y_true,y_pred):
    #kl = tf.keras.losses.KLDivergence()
    #return tf.losses.mean_squared_error(y_true,y_pred)+tf.losses.softmax_cross_entropy(y_true,y_pred)
    #return tf.losses.mean_squared_error(y_true,y_pred)+abs(kl(y_true,y_pred))
    #return tf.losses.huber_loss(y_true,y_pred,delta = 0.1)
    #y = (y_true+y_pred)/2
    #return tf.losses.huber_loss(y_true,y_pred,delta = 0.008)
    #return tf.keras.losses.MAE(y_true,y_pred)
    return tf.losses.mean_squared_error(y_true,y_pred)
    #return 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,max_val=1.0))
    #return 0.5*kl(y_true,y)+0.5*kl(y_pred,y) + tf.losses.huber_loss(y_true,y_pred,delta = 0.1)


def get_model(reload_model=True):
        if not reload_model:
            return load_model(Config.MODEL_PATH,custom_objects={'mse_ca':mse_ca})
        inp=Input((10,Config.SIZE1, Config.SIZE2,1))
        x = TimeDistributed(Conv2D(128, (7, 7), strides=1, padding="same"))(inp)
        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)
        x=TimeDistributed(Conv2D(64, (5, 5), strides=1, padding="same"))(x)
        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)
        x=TimeDistributed(Conv2D(64, (5, 5), strides=1, padding="same"))(x)
        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)

        x=ConvGRU2D(filters=32, kernel_size=(5,5),
                           padding='same', return_sequences=True,
                           )(x)

        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)

        x=ConvGRU2D(filters=16, kernel_size=(5,5),
                           padding='same', return_sequences=True,
                           )(x)

        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)

        x=ConvGRU2D(filters=32, kernel_size=(5,5),
                           padding='same', return_sequences=True,
                           )(x)

        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)
        x=TimeDistributed(Conv2DTranspose(64, (5, 5), strides=1, padding="same"))(x)
        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)

        x=TimeDistributed(Conv2DTranspose(64, (5, 5), strides=1, padding="same"))(x)
        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)

        x=TimeDistributed(Conv2DTranspose(128, (7, 7), strides=1, padding="same"))(x)
        x=BatchNormalization()(x)
        x=Activation('tanh')(x)
        x=Dropout(a)(x,training=True)
        out=Conv3D(filters=1, kernel_size=(1,1,1),
                       activation='sigmoid',
                       padding='same', data_format='channels_last')(x)

        seq=Model(inputs=inp, outputs=out)

        seq.compile(loss=mse_ca, optimizer= keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6),metrics=['accuracy'])
        seq.summary()





        tensorboard = TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=True)
        checkpoint = ModelCheckpoint(filepath=Config.MODEL_PATH,verbose=1,monitor='val_loss', save_weights_only=False,mode='auto' ,save_best_only=True,period=1)
        
        training_set = getdata(Config.DATASET_PATH)
        training_set = np.array(training_set)
        training_set = training_set.reshape(-1,Config.SEQENCE,Config.SIZE1,Config.SIZE2,1)
        
        testing_set = getdata(Config.VAL_PATH)
        testing_set = np.array(testing_set)
        testing_set = testing_set.reshape(-1,Config.SEQENCE,Config.SIZE1,Config.SIZE2,1)


        history=seq.fit(training_set, training_set, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, validation_data = (testing_set,testing_set), shuffle=False ,callbacks=[checkpoint,tensorboard]).history

        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)

        ax.plot(history['loss'], 'b', label='Train', linewidth=2)
        ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
        ax.set_title('Model loss', fontsize=16)
        ax.set_ylabel('Loss (mae)')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        np.savetxt(str(Config.DA)+'_train.txt',history['loss'])
        np.savetxt(str(Config.DA)+'_val.txt',history['val_loss'])
        plt.savefig(str(Config.DA)+'.png')
        plt.show()
        seq.save(Config.MODEL_PATH)
        return seq


def shift_data(data, n_frames=10):
    # noise_f = (-1)**np.random.randint(0, 2)
    # X = data[:, 0:n_frames, :, :, :]+noise_f*0.1
    X = data[:, 0:n_frames, :, :, :]
    y = data[:, n_frames:20, :, :, :]
    return X, y  # select a random observation


def shift_data2(data, n_frames=10):
    XX = data[1, 0:n_frames, :, :, :]
    yy = data[1, 1:n_frames + 1, :, :, :]
    return XX, yy  # select a random observation

model = get_model(Config.STATE)


def get_single_test():
    sz = Config.SZ
    test = np.zeros(shape=(sz, Config.SIZE1, Config.SIZE2, 1))
    cnt = 0
    for f in sorted(listdir(Config.SINGLE_TEST_PATH)):
        if str(join(Config.SINGLE_TEST_PATH, f))[-3:] == "jpg":
            image = cv.imread(join(Config.SINGLE_TEST_PATH, f))
            img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            img = cv.resize(img, (Config.SIZE1, Config.SIZE2))
            img = np.array(img[:, :, 2], dtype=np.uint8)
            img[img<70] = 0
            img = img / 255
            test[cnt, :, :, 0] = img
            cnt = cnt + 1
    return test


num = Config.TEST_NUM

def maxmin(data):

    value = np.zeros(data.shape[0])
    for ii in range(data.shape[0]):
        aa = np.zeros(10)
        
        for jj in range(10):
            aa[jj] = sum(data[ii,jj,:,:,].flatten())
        maxv = np.max(aa)
        minv = np.min(aa)
        value[ii] = (aa[0]-minv)/(maxv-minv)
    return value



def evaluate():
        print("get model")
        test = get_single_test()
        print("get test")
        sz = test.shape[0] - 20
        print(sz)
        sequences1 = np.zeros((sz, 10, Config.SIZE1, Config.SIZE2, 1))
        sequences2 = np.zeros((sz, 10, Config.SIZE1, Config.SIZE2, 1))
        # apply the sliding window technique to get the sequences
        
        for i in range(0, sz):
            clip = np.zeros((20, Config.SIZE1, Config.SIZE2, 1))
            for j in range(0, 20):
                clip[j,:,:,:] = test[i + j, :, :, :]
            sequences1[i] = clip[0:10,:,:,:]
            sequences2[i] = clip[10:20,:,:,:]
        

        # get the reconstruction cost of all the sequences
        reconstructed_sequences = model.predict(sequences1,batch_size=1)
        reconstructed_sequences1 = reconstructed_sequences.reshape(-1,10)

        sr = maxmin(reconstructed_sequences1)

        print(sequences1.shape)
        print(reconstructed_sequences[5].shape)
        
        
        sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences1[i],reconstructed_sequences[i])) for i in range(0,sz)])
        #print(scipy.stats.entropy(sequences1[0].flatten(), reconstructed_sequences[0].flatten()))
        #sequences_reconstruction_cost = np.array([np.sum(scipy.stats.entropy(sequences1[i].flatten(), reconstructed_sequences[i].flatten())) for i in range(0, sz)])
        print("形状",sequences_reconstruction_cost.shape)
        sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / (np.max(sequences_reconstruction_cost)-np.min(sequences_reconstruction_cost))
        

        # #sa = sequences_reconstruction_cost
        # print(1111111111111111111111111111111)
        # print(np.min(sequences_reconstruction_cost))
        # print(np.max(sequences_reconstruction_cost))
        # print(1111111111111111111111111111111)
        # #sr = 1.0 - sa


        print(sr)
        figsize = 11,9
        plt.rcParams['font.family']='SimHei'
        plt.rcParams['axes.unicode_minus']=False
        a = sr.reshape(sz,1)
        np.savetxt('41.txt',a)
        plt.plot(sr,linewidth=3.0,ms=10)
        plt.xticks(fontsize=15,weight='bold')
        plt.yticks(fontsize=15,weight='bold')
        #plt.ylabel('正常分数SN(t)')
        #plt.xlabel('红外成像序列时间 t/s')
        ax=plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        strname = Config.SINGLE_TEST_PATH
        str11 = strname.split('/')
        savefolder1 = str11[-1]+'_'+str(Config.DA)
        if not os.path.exists(savefolder1):
            os.mkdir(savefolder1)
        plt.savefig(savefolder1+"/"+'mse'+'.png')
        plt.show()
        MSEMATRIX = abs(sequences1 - reconstructed_sequences)
        print(np.max(MSEMATRIX[:,0,:,:,]))
        print(np.min(MSEMATRIX[:,0,:,:,]))
        print(MSEMATRIX.shape)
        MSEMATRIX = MSEMATRIX[:,:,:,:,0]
        strname = Config.SINGLE_TEST_PATH
        str11 = strname.split('/')
        savefolder2 =  savefolder1+'/'+'msematrix'
        os.mkdir(savefolder2)
        #levels = [0.005,0.01,0.05,0.1,0.15,0.2,0.5,0.8,0.9,1]
        levels = [0.005,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7]
        for iii in range(Config.SZ-20):
            IMG11 = MSEMATRIX[int(iii),0,:,:]
            #IMG11 = np.multiply(RELI_IMG[int(iii),0,:,:],MO)
            #IMG11 = RELI_IMG[int(iii),0,:,:] - MO
            # print(111111111111111111111111111)
            # print(np.max(IMG11))
            # print(np.min(IMG11))
            # print(111111111111111111111111111)
            #IMG11[IMG11<0.07]=0.05
            plt.ylim((64, 0))
            figsize = 11,9
            h = plt.contourf(IMG11,levels,cmap='jet')
            plt.colorbar()
            plt.savefig(savefolder2+"/"+str(iii)+'.png')
            plt.clf()

def evaluate_reli1():
    test = get_single_test()
    print(test[0,:,:,])
    print(np.max(test[0,:,:,]))
    print(np.min(test[0,:,:,]))
    plt.imshow(test[110,:,:,],cmap = 'gray')
    #plt.show()
    print("get test")
    sz = test.shape[0] - int(Config.SEQENCE*2) 




def evaluate_reli():
    test = get_single_test()
    print(test[0,:,:,])
    print(np.max(test[0,:,:,]))
    print(np.min(test[0,:,:,]))
    plt.imshow(test[110,:,:,],cmap = 'gray')
    #plt.show()
    print("get test")
    sz = test.shape[0] - int(Config.SEQENCE*2)
    
    RELI_MATRIX = np.zeros((num,sz,Config.SEQENCE, Config.SIZE1, Config.SIZE2, Config.CHANNEL))

    for ii in range(num):
        #model = get_model(Config.STATE)
        print("get model")
        print(sz)
        sequences1 = np.zeros((sz, Config.SEQENCE, Config.SIZE1, Config.SIZE2, Config.CHANNEL))
        sequences2 = np.zeros((sz, Config.SEQENCE, Config.SIZE1, Config.SIZE2, Config.CHANNEL))
        # apply the sliding window technique to get the sequences
        for i in range(0, sz):
            clip = np.zeros((int(Config.SEQENCE*2), Config.SIZE1, Config.SIZE2, Config.CHANNEL))
            for j in range(0, int(Config.SEQENCE*2)):
                clip[j,:,:,:] = test[i + j, :, :, :]
            sequences1[i] = clip[0:int(Config.SEQENCE),:,:,:]
            sequences2[i] = clip[int(Config.SEQENCE):int(Config.SEQENCE*2),:,:,:]

        # get the reconstruction cost of all the sequences
        reconstructed_sequences = model.predict(sequences1,batch_size=1)




        

        print(reconstructed_sequences.shape)
        print('jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj')

        MSE = (np.square(np.subtract(reconstructed_sequences[0,0,:,:,],sequences2[0,0,:,:,]))).mean()

        print(MSE)
        RELI_MATRIX[ii,:,:,:,:,:] = reconstructed_sequences
   


    plt.imshow(RELI_MATRIX[-1,110,0,:,:,],cmap='gray')
    #plt.show()
    plt.imshow(sequences1[110,0,:,:,],cmap='gray')
    #plt.show()
    RELI_IMG = RELI_MATRIX.reshape((num,-1))

    RELI_IMG = RELI_IMG/RELI_IMG.max(axis=0)

    RELI_IMG = np.std(RELI_IMG,axis=0)

    
    print('lllllllllllllllllllllllllllllllll')
    print(RELI_IMG.shape) 

    RELI_IMG = RELI_IMG.reshape((sz, Config.SEQENCE, Config.SIZE1, Config.SIZE2, Config.CHANNEL))

    RELI_IMG = RELI_IMG


    MO11 = RELI_IMG[:100,0,:,:,].reshape((100,-1))
    #sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences2[i],reconstructed_sequences[i])) for i in range(0,sz)])

    strname = Config.SINGLE_TEST_PATH
    str11 = strname.split('/')
    savefolder1 = str11[-1]+'_'+str(Config.DA) + '_' + str(Config.TEST_NUM)
    if not os.path.exists(savefolder1):
        os.mkdir(savefolder1)
    RELISUM = np.array([sum(RELI_IMG[i,0,:,:,].flatten()) for i in range(0,sz)])

    print(11111111111111111111111111111)

    print(np.max(RELISUM))
    print(np.min(RELISUM))
    print(RELISUM.shape)
    plt.plot(RELISUM,linewidth=3.0,ms=10)
    plt.savefig(savefolder1+"/"+'un1'+'.png')
    #plt.show()
    print(11111111111111111111111111111)

    sa = RELISUM/4096

    #sa = (RELISUM - np.min(RELISUM)) / (np.max(RELISUM)-np.min(RELISUM))

    #sa = maxmin(RELI_IMG)

    sr = 1-sa
    


    plt.plot(sr,linewidth=3.0,ms=10)
    plt.savefig(savefolder1+"/"+'un2'+'.png')
    #plt.show()
    strname = Config.SINGLE_TEST_PATH
    str11 = strname.split('/')
    rocname = savefolder1+ "/" + 'ROC'+str11[-1]+str(Config.DA)+'.txt'
    np.savetxt(rocname,sr)

    print("最大值：",np.max(RELI_IMG))
    print("最小值：",np.min(RELI_IMG))    

    RELI_IMG = RELI_IMG[:,:,:,:,0]
    #RELI_IMG[RELI_IMG>0.15] = 0.005
    
    MO = np.mean(MO11,axis=0)

    MO = MO.reshape((Config.SIZE1, Config.SIZE2))
    #MO = np.copy(RELI_IMG[0,0,:,:])


    # MO[MO>0.05]=0
    # MO[MO!=0]=1


    #RELI_IMG1 = (RELI_IMG - np.min(RELI_IMG))/(np.max(RELI_IMG)-np.min(RELI_IMG))

    RELI_IMG = RELI_IMG





    #levels = [0.005,0.01,0.05,0.1,0.15,0.2,0.5,0.8,1,1.2]
    levels = [0.00,0.005,0.01,0.05,0.07,0.1,0.2,0.3]
    #levels = [0.0005,0.005,0.01,0.05]
    #levels = [0,0.3,0.4,0.5,0.6]

    # fig, ax = plt.subplots(3,10)
    # ax = ax.flatten()
    # names = locals()

    ##############正常无泄漏数据##############
    #for jj1 in range(Config.SN):
    # for jjj in range(Config.SZ-20):
    #     IMG11 = RELI_IMG[int(jjj),0,:,:]









    strname = Config.SINGLE_TEST_PATH
    str11 = strname.split('/')
    savefolder2 = savefolder1 +'/'+ 'hotmap'
    os.mkdir(savefolder2)
    MO[MO<0.07] = 0

    for iii in range(Config.SZ-20):
        #IMG11 = RELI_IMG[int(iii),0,:,:]
        #IMG11 = np.multiply(RELI_IMG[int(iii),0,:,:],MO)
        IMG11 = RELI_IMG[int(iii),0,:,:] - MO
        # print(111111111111111111111111111)
        # print(np.max(IMG11))
        # print(np.min(IMG11))
        # print(111111111111111111111111111)
        if sum(IMG11.flatten()>0.1)<220:
            IMG11[IMG11<0.2]=0.05
        IMG11[IMG11<0.1]=0.05
        plt.ylim((64, 0))
        figsize = 11,9

        h = plt.contourf(IMG11,levels,cmap='jet')
        plt.colorbar()
        np.savetxt(savefolder2+"/"+str(iii)+'.txt',IMG11)
        plt.savefig(savefolder2+"/"+str(iii)+'.png')
        plt.clf()

    IMG11 = RELI_IMG[int(10),0,:,:]

    


    plt.subplot(1, 4, 1)

    #h = plt.imshow(IMG11,cmap='jet')
    plt.ylim((64, 0))
    figsize = 11,9
    h = plt.contourf(IMG11,levels,cmap='jet')
    plt.xticks(fontsize=15,weight='bold')
    plt.yticks(fontsize=15,weight='bold')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
        #ax[jj1+1].invert_yaxis()
        #h = ax[jj1+1].contourf(IMG11,levels,cmap='jet')


    ##############泄漏初期扩张################
    #for jj2 in range(Config.SN):
        
        

    IMG22 = RELI_IMG[int(110),0,:,:]

    plt.subplot(1, 4, 2)

    #h = plt.imshow(IMG22,cmap='jet')
    plt.ylim((64, 0))
    figsize = 11,9
    h = plt.contourf(IMG22,levels,cmap='jet')
    plt.xticks(fontsize=15,weight='bold')
    plt.yticks(fontsize=15,weight='bold')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)



        #ax[jj2+11].invert_yaxis()
        #h = ax[jj2+11].contourf(IMG22,levels,cmap='jet')


    ##############泄漏后期稳定################
    #for jj3 in range(Config.SN-3):
        
        

    IMG33 = RELI_IMG[int(160),0,:,:]

    plt.subplot(1, 4, 3)

    #h = plt.imshow(IMG33,cmap='jet')
    plt.ylim((64, 0))
    figsize = 11,9
    h = plt.contourf(IMG33,levels,cmap='jet')
    plt.xticks(fontsize=15,weight='bold')
    plt.yticks(fontsize=15,weight='bold')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

        #ax[jj3+21].invert_yaxis()
        #h = ax[jj3+21].contourf(IMG33,levels,cmap='jet')

    


    # fig.colorbar(h,ax=[ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6], ax[7], ax[8],ax[9], ax[10], ax[11], ax[12], ax[13], ax[14], ax[15], ax[16], ax[17],ax[18], ax[19], ax[20], ax[21], ax[22], ax[23], ax[24], ax[25], ax[26],ax[27], ax[28], ax[29]],fraction=0.03, pad=0.05)
    plt.xticks(fontsize=15,weight='bold')
    plt.yticks(fontsize=15,weight='bold')
    #plt.ylabel('正常分数SN(t)')
    #plt.xlabel('红外成像序列时间 t/s')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    # cb = plt.colorbar(h)

    # cb.ax.tick_params(labelsize=15,weight='bold')

    plt.show()





if Config.RELI:
    evaluate()
    #kmeas_imgs()
else:
    evaluate_reli()

