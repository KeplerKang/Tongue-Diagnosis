import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import warnings#引入警告信息库
warnings.filterwarnings('ignore')#过滤警告信息
from data import *
from sklearn.svm import SVR
#导入数据预处理工具
from sklearn.preprocessing import StandardScaler


X_train,y_train,X_test,y_test=train_pca[:1000],alllabels[:1000],test_pca[:100],testlabels[:100]
#打印训练集与测试集状态
print('train datasets size:',X_train.shape)
print('test datasets size:',y_train.shape)
print('test datasets size:',X_test.shape)
print('test datasets size:',y_test.shape)
print('\n')

# for kernel in ['linear','rbf','sigmoid']:
#     svr = SVR(kernel = kernel,gamma = 'auto')
#     svr.fit(X_train,y_train)
#     print(kernel,'核函数的模型训练集得分：{:.3f}'.format(svr.score(X_train,y_train)))
#     print(kernel,'核函数的模型测试集得分：{:.3f}\n'.format(svr.score(X_test,y_test)))
#对数据集进行数据预处理
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#用预处理后的数据查询训练模型
for kernel in ['linear','rbf','sigmoid']:
    svr = SVR(kernel = kernel)
    svr.fit(X_train,y_train)
    print('数据预处理后',kernel,'核函数的模型训练集得分：{:.3f}'.format(svr.score(X_train,y_train)))
    print('数据预处理后',kernel,'核函数的模型测试集得分：{:.3f}\n'.format(svr.score(X_test,y_test)))
svr = SVR(C = 100,gamma = 0.1,kernel = 'rbf')
svr.fit(X_train_scaled,y_train)
print('调节参数后的''rbf内核的SVR模型训练集得分：{:.3f}'.format(svr.score(X_train_scaled,y_train)))
print('调节参数后的''rbf内核的SVR模型测试集得分：{:.3f}\n'.format(svr.score(X_test_scaled,y_test)))