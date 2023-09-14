import os
import cv2
import random
import itertools
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm, logistic
from scipy import stats
from scipy.special import rel_entr
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='./font/timesbd.ttf',size=40,weight='light')
linewid = 4
tickl = 8



class Config:
    MODEL_PATH = './model/amone_0.1_1.h5'
    TRP = './datasets/train/all'
    TSP = './datasets/shiwai_2/88022'
    DPR = 0.1
    DPS = True
    BS1 = 550
    BS2 = 1173
    DSZ = (64,64,1)
    MSC = 200
    CLA = 2
    BSZ = 102
    EPS = 1000
    PAT = './uncer_img880_5'
    STA = False



SHAPE = Config.DSZ



def set_seed(seed):
    tf.random.set_random_seed(seed)
    #tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def get_model(train=True):
    
    #set_seed(33)
    
    # pre_process = Lambda(preprocess_input)
   
    # vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (64,64,3))
    # vgg = Model(vgg.input, vgg.layers[-3].output)
   
    
    # vgg.summary()
    
    # vgg._layers.pop(5)
    # vgg._layers.pop(4)
    # vgg._layers.pop(3)
    # vgg._layers.pop(2)
    # vgg._layers.pop(1)
    #vgg._layers.pop(0)


    #vgg.summary()

    #gg.trainable = False

    inp = Input(SHAPE)
    
    xxinp = Reshape((4096,1))(inp)

    attention_probs = Dense(1,activation='sigmoid',name='attention_vec1')(xxinp)
    xxx = Multiply()([xxinp,attention_probs])
    xxx = Reshape(SHAPE)(xxx)



    vgg_16_process = GaussianNoise(0.02)(xxx)
    #print(inp.shape)



    out = Convolution2D(filters=64,kernel_size=3,activation='relu',strides=1,padding='same')(vgg_16_process)
    



    # out = Dropout(Config.DPR)(out,training=Config.DPS)
    # out = BatchNormalization()(out)
    # out = Convolution2D(filters=64,kernel_size=3,activation='relu',strides=1,padding='same')(out)
    # out = Dropout(Config.DPR)(out,training=Config.DPS)
    
    # out = BatchNormalization()(out)
    # out = MaxPooling2D((2,2),strides=2)(out)
    # out = Convolution2D(filters=128,kernel_size=3,activation='relu',strides=1,padding='same')(out)
    # out = Dropout(Config.DPR)(out,training=Config.DPS)
    # out = BatchNormalization()(out)
    # out = Convolution2D(filters=128,kernel_size=3,activation='relu',strides=1,padding='same')(out)
    # out = Dropout(Config.DPR)(out,training=Config.DPS)
    # out = BatchNormalization()(out)
    # out = MaxPooling2D((2,2),strides=2)(out)
    # out = Convolution2D(filters=256,kernel_size=15,activation='relu',strides=1,padding='same')(out)
    # out = MaxPooling2D()(out)
    # out = Convolution2D(filters=256,kernel_size=15,activation='relu',strides=1,padding='same')(out)
    # out = MaxPooling2D()(out)
    # out = Convolution2D(filters=512,kernel_size=15,activation='relu',strides=1,padding='same')(out)
    # out = MaxPooling2D()(out)
    # out = Convolution2D(filters=512,kernel_size=15,activation='relu',strides=1,padding='same')(out)
    # out = MaxPooling2D()(out)
    vgg_out = Flatten()(out)

    #vgg_out = vgg(inp)

    #vgg_out = Flatten()(vgg_out)
    
    print(55555555555555555555)
    print(vgg_out.shape)
    print(55555555555555555555)
    #noise = np.zeros((102,32768),dtype=np.float32)
    #noise = noise.reshape(-1,32768)
    print(222222222222222222222)
    #print(noise.shape)
    print(222222222222222222222)
    # noise = Lambda(tf.zeros_like)(vgg_out)
    # noise=tf.cast(noise,dtype=tf.int32)
    # noise = tf.random.normal([32768],0,0.1)
    
    noise = Lambda(tf.zeros_like)(vgg_out)
    noise = GaussianNoise(0.03)(noise)

    if train:
        x = Lambda(lambda z: tf.concat(z, axis=0))([vgg_out,noise])
        x = Activation('relu')(x)
    else:
        x = vgg_out
        
    # x = Dense(128, activation='relu')(x)
    x = Dropout(Config.DPR)(x,training=Config.DPS)
    x = Dense(64, activation='relu')(x)
    x = Dropout(Config.DPR)(x,training=Config.DPS)
    out1 = Dense(2, activation='softmax')(x)

    print(88888888888888888)
    print(out.shape)
    print(88888888888888888)

    model = Model(inp, out1)
    model.compile(Adam(lr=1e-4), loss='binary_crossentropy')
    model.summary()
    return model





def showdata(data):
    for ii in range(0,data.shape[0]):
        plt.imshow(data[ii,:,:,0])
        plt.show()


def getdata(paths,shapes):
    filelist = os.listdir(paths)
    traindata = np.empty((len(filelist),shapes[0]*shapes[1]*shapes[2]))

    n = 0
    
    filelist.sort(key=lambda x:int(x[:-4]))
    for file in filelist:
        if file.split('.')[1]=='txt':
            path = os.path.join(paths,file)
            imgdata = np.zeros((shapes[0], shapes[1],shapes[2]))
            data = np.loadtxt(path)
            
            imgdata[:,:,0] = data.reshape((shapes[0], shapes[1]))
            # imgdata[:,:,1] = data.reshape((shapes[0], shapes[1]))
            # imgdata[:,:,2] = data.reshape((shapes[0], shapes[1]))

            traindata[n,:] = imgdata.reshape(shapes[0]*shapes[1]*shapes[2])
            n = n + 1
    x = traindata.reshape((-1,shapes[0],shapes[1],shapes[2]))
    lables = np.zeros((len(filelist)))
    y=gggg(lables)
    return x,y




def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
### DEFINE SOME PARAMETERS ###


def gggg(y):
    y = np.zeros((len(y),2))
    y[:,0] = 1
    # zeros = np.zeros((y.shape[0],y.shape[1]))
    # zeros[:,0] = 0
    # zeros[:,1] = 1
    print(y)
    #y = np.concatenate([y,zeros],axis=0)
    #y = tf.convert_to_tensor(y)

    print(7777777777777777777777)
    print(y.shape)
    return y

def analyse_model_prediction(model,image11, label = None, forward_passes = Config.MSC,labelnum=Config.CLA,iinum=0):
    
    aa = image11.reshape((-1,SHAPE[0],SHAPE[1],SHAPE[2]))
    bb = np.zeros(aa.shape)

   # bb[:,:20,:,:] = aa[:,44:,:,:]
    bb[:,10:,:,:] = aa[:,:54,:,:] 

    image = bb

    if label is not None:
        label = np.argmax(label, axis = -1)
    
    extracted_probabilities = np.empty(shape=(forward_passes, labelnum))
    #extracted_std = np.empty(shape=(forward_passes, 10))
    for i in range(0,forward_passes):
        model_output_distribution = model.predict(image) 
        
        extracted_probabilities[i] = model_output_distribution.flatten()
        
        #extracted_std[i] = model_output_distribution.stddev().numpy().flatten()

    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    #plt.xticks(fontsize = 16, rotation = 45)
    #plt.yticks(fontsize = 16)

    # Show the image and the true label if provided.

    # Obtain the 95% prediction interval.
    # extracted_probabilities.shape = (forward_passes, 10)
    # So if we sample from the model 100 times, there will be 100 different
    # values for each of the 10 classes. 
    # We get the interval for each of the classes independently.

    std1 = np.std(extracted_probabilities[:, 0])
    max1 = np.max(extracted_probabilities[:, 0])
    std2 = np.std(extracted_probabilities[:, 1])
    max2 = np.max(extracted_probabilities[:, 1])
    print(std1,max1,std2,max2)

    print(sum(rel_entr(extracted_probabilities[:, 0],extracted_probabilities[:, 1])))

    # ax1.imshow(image.squeeze(), cmap='gray')
    # ax1.axis('off')
    # if label is not None:
    #     ax1.set_title('True Label: {}'.format(str(label)), fontsize = 20)
    # else:
    #     ax1.set_title('True Label Not Given', fontsize = 20)
    
    
    pct_2p5 = np.array([np.percentile(extracted_probabilities[:, i], 
                                      2.5) for i in range(labelnum)])
    pct_97p5 = np.array([np.percentile(extracted_probabilities[:, i], 
                                       97.5) for i in range(labelnum)]) 

    # Std also contains 100 different values. We take median across the column
    # to obtain a single value for each of the class label.
    # extracted_std = np.median(extracted_std, axis = 0)
    # highest_var_label = np.argmax(extracted_std, axis = -1)
    # if label is not None:
    #     print('Label %d has the highest std in this'
    #     ' prediction with the value %.3f' %(highest_var_label,
    #                                         extracted_std[highest_var_label]))
    # else:
    #   print('Std Array:', extracted_std)     
    
    # bar = ax2.bar(np.arange(labelnum), pct_97p5, color='red')
    # if label is not None:
    #     bar[int(label)].set_color('green')
    # # for x,y,z in zip(np.arange(labelnum),pct_97p5,[std1])
    # # ax2.text(np.arange(labelnum),pct_97p5[:,0],std1,ha='center',va='bottom')

    # # ax2.text(np.arange(labelnum),pct_97p5[:,1],std2,ha='center',va='bottom')
    



    # ax2.bar(np.arange(labelnum), pct_2p5-0.02, color='white', 
    #         linewidth=4, edgecolor='white')
    # ax2.set_xticks(np.arange(labelnum))
    
    # ax2.set_ylim([0, 1])
    # ax2.set_ylabel('Probability', fontsize = 18)
    # ax2.set_title('klvalue: {}'.format(str(sum(rel_entr(extracted_probabilities[:, 0],extracted_probabilities[:, 1])))), fontsize = 20)
    data_50=extracted_probabilities[:, 0]
    data_50=data_50.reshape(200,1)
    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
    print(data_50.shape)
    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
    x=np.linspace(0,1,50)
    mean_50=np.mean(data_50[:])
    std_50=np.std(data_50[:])
    print(np.mean(data_50[:]))
    print(np.std(data_50[:]))
    #mean_50,std_50=norm.fit(data_50[:,4])
    y_50=norm.pdf(x,mean_50,std_50)
    #ax2.hist(data_50[:],bins=6,density=True,facecolor='blue',cumulative=False)
    ax2.plot(x,y_50,'b--',linewidth=linewid,label='Plume existed')

    data_51=extracted_probabilities[:, 1]
    data_51=data_51.reshape(200,1)
    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
    print(data_51.shape)
    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
    x1=np.linspace(0,1,50)
    mean_51=np.mean(data_51[:])
    print(np.mean(data_51[:]))
    print(np.std(data_51[:]))
    print(data_51[:])
    mean_51,std_51=norm.fit(data_51[:])
    y_51=norm.pdf(x1,mean_51,std_51)
    ax2.plot(x1,y_51,'r--',linewidth=linewid,label='No released plume')

    ax2.set_xlim(0,1)
    a0 = y_50.max()
    a1 = y_51.max()
    
    if a0 > a1:
        a = a0
    else:
        a = a1

    ax2.set_ylim(0,a+int(1.2*a/3))

    ax2.set_title('KL_Dis: {}'.format(str(round(sum(rel_entr(extracted_probabilities[:, 0],extracted_probabilities[:, 1])),2))), fontproperties=myfont)
    #ax2.set_ylim(-0.1,1.1)
    ax2.legend(loc=1,prop=myfont)
    ax2.tick_params(bottom=True,axis='x',direction='out',which='major',length=tickl,width=linewid,color='k')
    ax2.tick_params(left=True,axis='y',direction='out',which='major',length=tickl,width=linewid,color='k')
    ax2.patch.set_facecolor('white')
    ax2.spines['top'].set_color('black')
    ax2.spines['top'].set_visible(True)
    ax2.spines['top'].set_linewidth(linewid)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['bottom'].set_visible(True)

    ax2.spines['bottom'].set_linewidth(linewid)
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_linewidth(linewid)
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_visible(True)
    ax2.spines['left'].set_linewidth(linewid)



    for axlab in ax2.get_xticklabels()+ax2.get_yticklabels():
        
        axlab.set_fontproperties(myfont)

    #plt.show()
    imgpath = os.path.join(Config.PAT,str(iinum)+'.jpg')
    plt.savefig(imgpath,bbox_inches='tight',dpi=720)

def batch_generator(data, bs):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=bs, replace=False)
        print(888888888888888)
        print(mini_batch_indices)
        xx = data[0][mini_batch_indices,:,:]
        yy = data[1][mini_batch_indices,:]
        zeros = np.zeros((yy.shape[0],yy.shape[1]))
        zeros[:,0] = 0
        zeros[:,1] = 1
        yy = np.concatenate([yy,zeros],axis=0)
        #tbr = [data for k in mini_batch_indices]
        yy = tf.convert_to_tensor(yy)
        yield xx,yy




trainx,trainy = getdata(Config.TRP,SHAPE)

trainx[trainx>0.0001] = 1
trainx[trainx==0.0001] = 1
trainx[trainx<0.0001] = 0
# trainx[trainx>0.07] = 1
# trainx[trainx==0.07] = 1
# trainx[trainx<0.07] = 0
#showdata(trainx)

#showdata(trainx)
#trainx = trainx/255
#trainx = np.dstack([trainx]*3)
print(trainx.shape)



testx,testy = getdata(Config.TSP,SHAPE)
#testx = np.dstack([testx]*3)


#testx = testx/255

print(11111111111111)
print(np.max(testx))
print(np.min(testx))
print(44444444444444)



# testx[testx>0.0001] = 1
# testx[testx==0.0001] = 1
# testx[testx<0.0001] = 0


testx[testx>0.07] = 1
testx[testx==0.07] = 1
testx[testx<0.07] = 0
#showdata(testx)
# testx[testx==255] = 1
# testx[testx==0] = 0


if Config.STA:
    model = get_model()


    #s_batch = batch_generator([xs_train1,ys_train],Config.BS)


    #xs,ys = next(s_batch)
    tensorboard = TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=True)
    checkpoint = ModelCheckpoint(filepath=Config.MODEL_PATH,verbose=1,monitor='loss',save_weights_only=False,mode='auto' ,save_best_only=True,period=1)

    #es = EarlyStopping(monitor='loss', mode='auto', restore_best_weights=True, verbose=1, patience=5)
    print(batch_generator([trainx,trainy],Config.BSZ))
    xx,yy = next(batch_generator([trainx,trainy],Config.BSZ))
    
    history=model.fit(xx, yy, epochs=Config.EPS,  shuffle=False, steps_per_epoch=20,callbacks=[checkpoint,tensorboard]).history
    #history=seq.fit(training_set, training_set, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, validation_data = (testing_set,testing_set), shuffle=False ,callbacks=[checkpoint,tensorboard]).history
    model.save_weights("weights.h5")
    model.save_weights("weights1.h5")
    model.save_weights("weights2.h5")
    model.save_weights("weights3.h5")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)

    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    #ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    # np.savetxt(str(Config.DA)+'_train.txt',history['loss'])
    # np.savetxt(str(Config.DA)+'_val.txt',history['val_loss'])
    # plt.savefig(str(Config.DA)+'.png')
    plt.show()

else:
    #model2 = load_model(Config.MODEL_PATH)
    
    model2 = get_model(train=False)
    model2.load_weights(Config.MODEL_PATH)
    #model2.set_weights(model.get_weights())
    #model2=tf.keras.models.load_model('./model/model_1.hdf5')
    print('1')
    #model = load_model(Config.MODEL_PATH)




iinum = 0
#testy = np.loadtxt('926label.txt')
for iii in range(0,Config.BS2):

    analyse_model_prediction(model2,testx[iii,:],testy[iii],Config.MSC,Config.CLA,iinum)

    iinum = iinum + 1



xx=model2.predict(testx)
pred_test = np.argmax(xx, axis=1)
# np.savetxt(Config.NA1,xx[:int(Config.BS2),])
# np.savetxt(Config.NA2,pred_test[:int(Config.BS2)])

print(xx[0:Config.BS2,:])
print(pred_test[0:Config.BS2])
#print('ACCURACY:', accuracy_score(testy, pred_test))

#cnf_matrix = confusion_matrix(testy, pred_test)
plt.figure(figsize=(7,7))
#plot_confusion_matrix(cnf_matrix, classes=['not CAT','CAT'])
#plt.show()
