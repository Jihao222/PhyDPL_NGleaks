#Importing Required Modules
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import rcParams
from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,confusion_matrix
from sklearn import metrics


plt.rcParams['xtick.direction'] = 'in'
plt.rc('xtick.major',width='2',size='6')
plt.rcParams['ytick.direction'] = 'in'
plt.rc('ytick.major',width='2',size='6')
plt.rcParams['axes.linewidth'] = 2  # 图框宽度
#plt.rcParams['figure.dpi'] = 300  # plt.show显示分辨率
font1 = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 18}
font2 = {'family': 'simsun',
         'weight': 'bold',
         'size': 18,
         }




plt.rc('font', **font1)
plt.rcParams['figure.autolayout'] = True 
fig = plt.figure(figsize=(10,8),dpi=80)
markes = ['-o', '-s', '-^', '-p', '-^', '-v', '-p', '-d', '-h', '-2', '-8', '-6']
# ax = plt.axes()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

def tsne(data):
        data[515:780] =  0.9
        data[1405:1560] =  0.9
        data[2195:2340] = 0.9
        data[2896:3120] = 0.9
        return data 






y_test = np.loadtxt('./shiwai_all/shiwai_real_label.txt')

print('zzzzzzzzzzzzzzzzzzzzzzz')
print(y_test.shape)
print('zzzzzzzzzzzzzzzzzzzzzzz')

aaa = y_test

y_test1 = 1-y_test




probs1 = np.loadtxt('./shiwai_all/un1shiwai_all.txt')

print('zzzzzzzzzzzzzzzzzzzzzzz')
print(probs1.shape)
print('zzzzzzzzzzzzzzzzzzzzzzz')
probs1[probs1>0]=1
probs1[probs1==0]=0
probs1[probs1<0]=0
#probs1[probs1==0]=0




#probs1 = probs1[:,1]
#probs1 = tsne(probs1)
fpr1, tpr1, thresholds1 = roc_curve(y_test1, probs1)
print(thresholds1)
aucvalue1 = round(roc_auc_score(y_test1, probs1),4)
#print(roc_auc_score(y_test, probs1),metrics.auc(fpr1, tpr1))
#plt.scatter(fpr1,tpr1,c='k',alpha=1,marker='s')
plt.plot(fpr1, tpr1, markes[0], label='unceral1, AUC = ' + str(aucvalue1),linewidth=3)
f1_score,recall_score,precision_score
print('#############################uncer1##################################')
print('auc',round(roc_auc_score(y_test1, probs1),4))
print('f1',f1_score(y_test1, probs1,average='binary',pos_label=0))
print('rec',recall_score(y_test1, probs1,average='binary',pos_label=0))
print('prec',precision_score(y_test1, probs1,average='binary',pos_label=0))
print('prec',precision_score(y_test1, probs1))
print('prec',precision_score(y_test1, probs1,average=None))
print(confusion_matrix(y_test1, probs1))
print('#############################uncer1##################################')
#plt.plot(fpr1, tpr1, markes[0], label='MCMC抽样次数：5, AUC = ' + str(aucvalue1),linewidth=3)


probs11 = np.loadtxt('./shiwai_all/un0shiwai_all.txt')
#probs2 = probs2[:,1]
#probs2 = tsne(probs2)
fpr11, tpr11, thresholds11 = roc_curve(y_test, probs11)
print(thresholds11)
aucvalue11 = round(roc_auc_score(y_test, probs11),4)
#plt.scatter(fpr2,tpr2,c='k',alpha=1,marker='s')
plt.plot(fpr11, tpr11, markes[1], label='uncer0, AUC = ' + str(aucvalue11),linewidth=3)
#plt.plot(fpr2, tpr2, markes[1], label='MCMC抽样次数：10, AUC = ' + str(aucvalue2),linewidth=3)
print('#############################uncer0##################################')
print('auc',round(roc_auc_score(y_test, probs11),4))
print('f1',f1_score(y_test, probs11,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs11,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs11,average='binary',pos_label=0,zero_division='warn'))
print('prec',precision_score(y_test, probs11))
print('prec',precision_score(y_test, probs11,average=None))
print(confusion_matrix(y_test, probs11))
print('####################################################################')

probs12 = np.loadtxt('./shiwai_all/dp0shiwai_all.txt')
#probs2 = probs2[:,1]
#probs2 = tsne(probs2)
fpr12, tpr12, thresholds12 = roc_curve(y_test, probs12)
print(thresholds12)
aucvalue12 = round(roc_auc_score(y_test, probs12),4)
#plt.scatter(fpr2,tpr2,c='k',alpha=1,marker='s')
plt.plot(fpr12, tpr12, markes[1], label='dp0, AUC = ' + str(aucvalue12),linewidth=3)
#plt.plot(fpr2, tpr2, markes[1], label='MCMC抽样次数：10, AUC = ' + str(aucvalue2),linewidth=3)
print('#############################dp0##################################')
print('auc',round(roc_auc_score(y_test, probs12),4))
print('f1',f1_score(y_test, probs12,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs12,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs12,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs12))
print('prec',precision_score(y_test, probs12,average=None))
print('####################################################################')


probs13 = np.loadtxt('./shiwai_all/dp1shiwai_all.txt')
probs13[probs13>0]=1
probs13[probs13==0]=0
probs13[probs13<0]=0
fpr13, tpr13, thresholds13 = roc_curve(y_test1, probs13)
print(thresholds13)
aucvalue13 = round(roc_auc_score(y_test1, probs13),4)
#plt.scatter(fpr2,tpr2,c='k',alpha=1,marker='s')
plt.plot(fpr13, tpr13, markes[1], label='dp1, AUC = ' + str(aucvalue13),linewidth=3)
#plt.plot(fpr2, tpr2, markes[1], label='MCMC抽样次数：10, AUC = ' + str(aucvalue2),linewidth=3)
print('#############################dp1##################################')
print('auc',round(roc_auc_score(y_test1, probs13),4))
print('f1',f1_score(y_test1, probs13,average='binary',pos_label=0))
print('rec',recall_score(y_test1, probs13,average='binary',pos_label=0))
print('prec',precision_score(y_test1, probs13,average='binary',pos_label=0))
print('####################################################################')



probs2 = np.loadtxt('./shiwai_all/gmmshiwai_all.txt')
#probs2 = probs2[:,1]
#probs2 = tsne(probs2)
fpr2, tpr2, thresholds2 = roc_curve(y_test, probs2)
print(thresholds2)
aucvalue2 = round(roc_auc_score(y_test, probs2),4)
#plt.scatter(fpr2,tpr2,c='k',alpha=1,marker='s')
plt.plot(fpr2, tpr2, markes[1], label='gmm, AUC = ' + str(aucvalue2),linewidth=3)
#plt.plot(fpr2, tpr2, markes[1], label='MCMC抽样次数：10, AUC = ' + str(aucvalue2),linewidth=3)
print('#############################gmm##################################')
print('auc',round(roc_auc_score(y_test, probs2),4))
print('f1',f1_score(y_test, probs2,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs2,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs2,average='binary',pos_label=0))
print('####################################################################')


probs3 = np.loadtxt('./shiwai_all/we0shiwai_all.txt')
#probs3 = probs3[:,1]
#probs3 = tsne(probs3)
fpr3, tpr3, thresholds3 = roc_curve(aaa, probs3)
print(thresholds3)
aucvalue3 = round(roc_auc_score(aaa, probs3),4)
# plt.scatter(fpr3,tpr3,c='k',alpha=1,marker='s')
plt.plot(fpr3, tpr3, markes[2], label='wesambe0, AUC = ' + str(aucvalue3),linewidth=3)
#plt.plot(fpr3, tpr3, markes[2], label='MCMC抽样次数：15, AUC = ' + str(aucvalue3),linewidth=3)
print('#############################wesambe0##################################')
print('auc',round(roc_auc_score(y_test, probs3),4))
print('f1',f1_score(y_test, probs3,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs3,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs3,average='binary',pos_label=0))
print('####################################################################')

probs31 = np.loadtxt('./shiwai_all/we1shiwai_all.txt')
probs31[probs31>0]=1
probs31[probs31==0]=0
probs31[probs31<0]=0

#probs3 = probs3[:,1]
#probs3 = tsne(probs3)
fpr31, tpr31, thresholds31 = roc_curve(y_test1, probs31)
print(thresholds31)
aucvalue31 = round(roc_auc_score(y_test1, probs31),4)
# plt.scatter(fpr3,tpr3,c='k',alpha=1,marker='s')
plt.plot(fpr31, tpr31, markes[2], label='wesambe1, AUC = ' + str(aucvalue31),linewidth=3)
#plt.plot(fpr3, tpr3, markes[2], label='MCMC抽样次数：15, AUC = ' + str(aucvalue3),linewidth=3)
print('#############################wesambe##################################')
print('auc',round(roc_auc_score(y_test1, probs31),4))
print('f1',f1_score(y_test1, probs31,average='binary',pos_label=0))
print('rec',recall_score(y_test1, probs31,average='binary',pos_label=0))
print('prec',precision_score(y_test1, probs31,average='binary',pos_label=0))
print('####################################################################')


probs4 = np.loadtxt('./shiwai_all/cntshiwai_all.txt')
#probs4 = tsne(probs4)
fpr4, tpr4, thresholds4 = roc_curve(y_test, probs4)
print(thresholds4)
aucvalue4 = round(roc_auc_score(y_test, probs4),4)
# plt.scatter(fpr4,tpr4,c='k',alpha=1,marker='s')
plt.plot(fpr4, tpr4, markes[3], label='cnt, AUC = ' + str(aucvalue4),linewidth=3)
#plt.plot(fpr4, tpr4, markes[3], label='MCMC抽样次数：20, AUC = ' + str(aucvalue4),linewidth=3)
print('#############################cnt##################################')
print('auc',round(roc_auc_score(y_test, probs4),4))
print('f1',f1_score(y_test, probs4,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs4,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs4,average='binary',pos_label=0))
print('####################################################################')



probs5 = np.loadtxt('./shiwai_all/gmgshiwai_all.txt')
#probs5 = tsne(probs5)
fpr5, tpr5, thresholds5 = roc_curve(y_test, probs5)
print(thresholds5)
aucvalue5 = round(roc_auc_score(y_test, probs5),4)
# # plt.scatter(fpr5,tpr5,c='k',alpha=1,marker='s')
plt.plot(fpr5, tpr5, markes[4], label='gmg, AUC = ' + str(aucvalue5),linewidth=3)
print('#############################gmg##################################')
print('auc',round(roc_auc_score(y_test, probs5),4))
print('f1',f1_score(y_test, probs5,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs5,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs5,average='binary',pos_label=0))
print('####################################################################')




probs6 = np.loadtxt('./shiwai_all/knnshiwai_all.txt')
#probs5 = tsne(probs5)
fpr6, tpr6, thresholds6 = roc_curve(y_test, probs6)
print(thresholds6)
aucvalue6 = round(roc_auc_score(y_test, probs6),4)
# # plt.scatter(fpr5,tpr5,c='k',alpha=1,marker='s')
plt.plot(fpr6, tpr6, markes[5], label='knn, AUC = ' + str(aucvalue6),linewidth=3)
print('#############################knn##################################')
print('auc',round(roc_auc_score(y_test, probs6),4))
print('f1',f1_score(y_test, probs6,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs6,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs6,average='binary',pos_label=0))
print('####################################################################')





probs7 = np.loadtxt('./shiwai_all/mogshiwai_all.txt')
#probs5 = tsne(probs5)
fpr7, tpr7, thresholds7 = roc_curve(y_test, probs7)
print(thresholds7)
aucvalue7 = round(roc_auc_score(y_test, probs7),4)
# # plt.scatter(fpr5,tpr5,c='k',alpha=1,marker='s')
plt.plot(fpr7, tpr7, markes[6], label='mog, AUC = ' + str(aucvalue7),linewidth=3)
print('#############################mog##################################')
print('auc',round(roc_auc_score(y_test, probs7),4))
print('f1',f1_score(y_test, probs7,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs7,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs7,average='binary',pos_label=0))
print('####################################################################')




probs8 = np.loadtxt('./shiwai_all/m2mshiwai_all.txt')
#probs5 = tsne(probs5)
fpr8, tpr8, thresholds8 = roc_curve(y_test, probs8)
print(thresholds8)
aucvalue8 = round(roc_auc_score(y_test, probs8),4)
# # plt.scatter(fpr5,tpr5,c='k',alpha=1,marker='s')
plt.plot(fpr8, tpr8, markes[7], label='mog2m, AUC = ' + str(aucvalue8),linewidth=3)
print('#############################mog2m##################################')
print('auc',round(roc_auc_score(y_test, probs8),4))
print('f1',f1_score(y_test, probs8,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs8,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs8,average='binary',pos_label=0))
print('####################################################################')


probs9 = np.loadtxt('./shiwai_all/grashiwai_all.txt')
#probs5 = tsne(probs5)
fpr9, tpr9, thresholds9 = roc_curve(y_test, probs9)
print(thresholds9)
aucvalue9 = round(roc_auc_score(y_test, probs9),4)
# # plt.scatter(fpr5,tpr5,c='k',alpha=1,marker='s')
plt.plot(fpr9, tpr9, markes[7], label='grasta, AUC = ' + str(aucvalue9),linewidth=3)
print('#############################grasta##################################')
print('auc',round(roc_auc_score(y_test, probs9),4))
print('f1',f1_score(y_test, probs9,average='binary',pos_label=0))
print('rec',recall_score(y_test, probs9,average='binary',pos_label=0))
print('prec',precision_score(y_test, probs9,average='binary',pos_label=0))
print('####################################################################')


plt.xlabel('假阳率(FPR)',font2, weight='bold')
plt.ylabel('真阳率(TPR)',font2, weight='bold')
plt.xlim(0,1)
plt.ylim(0,1)

plt.legend(loc='best',prop=font2)
plt.show()

