# -*- coding: utf-8 -*-
# @Author : LuoXianan
# @File : __init__.py.py
# @Project: graduation design
# @CreateTime : 2022/5/15 16:33:59

#  首先导入数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizer_v2.adadelta import Adadelta
from tensorflow.python.keras.optimizer_v2.adagrad import Adagrad
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.saving.saved_model.load import metrics


dataset = pd.read_csv('data.csv')
print(dataset)

# print('查看数据前五行:')
# print(dataset.head())
# print('-'*40)
# 使用seaborn对数据进行观察
'''
seaborn库的简介
Seabn是一个在Python中制作有吸引力和丰富信息的统计图形的库。它构建在MatPultLB的顶部，
与PyDATA栈紧密集成，包括对SIMPY和BANDA数据结构的支持以及SISPY和STATSMODEL的统计例程。
Seaborn 其实是在matplotlib的基础上进行了更高级的 API 封装，从而使得作图更加容易 在大多数情况
下使用seaborn就能做出很具有吸引力的图，而使用matplotlib就能制作具有更多特色的图。应该把Seaborn
视为matplotlib的补充。Seabn是基于MatPultLB的Python可视化库。它为绘制有吸引力的统计图形提供了一个高级接口。
'''
import seaborn as sns

sns.pairplot(dataset.iloc[:, 1:6], hue='class')
plt.show()   # 显示图片


# 生成测试数据
# 将前4列与第5列分别抽离成np array
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:,4].values

# 打印分离出的特征与类别
print(X)
print('-'*40)
print(y)

# 将y字符串数组转换成整数数组，在这里我们可以使用sklearn的LabelEncoder库
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)

print(len(y1))
print(y1)
print('-'*40)

# 将y1转成神经网络需要的数组结构
Y = pd.get_dummies(y1).values

print('Y',len(Y))
print(Y)

# 将训练数据与测试数据做分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
# 训练集为30个
# print('X_train',len(X_train),X_train)
print('X_test',len(X_test),X_test)
# print('y_train',len(y_train),y_train)
print('y_test',len(y_test),y_test)


#  创建神经网络模型
'''
使用Sequential创建神经网络模型
模型一共4层
损失函数使用‘categorical_crossentropy’（比较适用于3种以上的分类的情况）
指定 metrics=[‘accuracy’]，会在训练结束后计算训练数据在模型上的准确率
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 代码创建一个Sequential模型，这里使用了一个采用线性激活的全连接(Dense)层。它实际上封装了输入值x乘以权重w，
# 加上偏置(bias)b，然后进行线性激活以产生输出。
'''
使用Sequential创建神经网络模型
模型一共4层
损失函数使用‘categorical_crossentropy’（比较适用于3种以上的分类的情况）
指定 metrics=[‘accuracy’]，会在训练结束后计算训练数据在模型上的准确率
'''
'''
# 对于具有 10 个类的单输入模型（多分类分类）：
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
model = Sequential()   # 创建序列模型
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='softmax'))
# 优化器 optimizer    损失函数 loss   评估标准 metrics
# 默认 Adam(learning_rate=0.04)
# SGD(lr=0.04, momentum=0.9, decay=0.0, nesterov=True)  都为accuracy = 0.42
# RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)  nerul 0.44 ,data 0.73
# Adagrad(lr=0.01, epsilon=None, decay=0.0)   nerul 0.54 ,data 0.73
# Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)  nerul 0.42 ,data 0.73
# Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) nerul 0.52,data 0.73
# Adamax(lr=0.009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) nerul 0.42,data 0.73
# Nadam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004) nerul 0.42,data 0.73
model.compile(Nadam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
              'CategoricalCrossentropy',
              metrics=['accuracy'])
# model.compile(optimizer='sgd(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)',
#               loss='categorical_crossentropy',
#               metrics=[metrics.accuracy])
model.summary()
'''
一行前面的 大写的字母 'I' 代表了Information，指的是一些提示性质的log信息。
你仔细看，前面那一大堆文本输出log，也都是I。所以这里这个提示，并不是导致你程序出错退出的原因。
MLIR 被用作实现和优化 Tensorflow 逻辑的另一种解决方案。此信息性消息是良性的，
表示未使用 MLIR。这是预期的，因为在 TF 2.3 中，基于 MLIR 的实现仍在开发和验证中
，因此最终用户通常不希望使用 MLIR 实现，而是希望使用非 MLIR 功能完整实现。
'''


# 训练模型
# 指定epochs=100，训练数据会在模型中训练100次
model.fit(X_train, y_train, epochs=100)


# 使用模型进行预测
y_pred = model.predict(X_test)
print(len(y_pred))
print(y_pred)

# 浮点类型的数据不方便理解，所以使用np.argmax将数据转为整数数组
y_pred_class = np.argmax(y_pred, axis=1)   # 其实就是记录每个数组中值最大的数的index
print('y_pred_class',len(y_pred_class))
print('y_pred_class',y_pred_class)

print('-'*40)

y_test_class = np.argmax(y_test,axis=1)
print('y_test_class',len(y_test_class))
print('y_test_class',y_test_class)

print('-'*40)

'''
这个错误是在使用如下包才出现的。
from sklearn.metrics import classification_report,accuracy_score
正如报错所说，你的模型的分类结果中有一类是没有被预测的，拿2分类来说，你的模型全部预测成了1或者0，就会报上述错误。例如：
实际标签 1，1，1，1，1，1，0，0
预测标签 0，0，0，0，0，0，0，0
UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in 
labels with no predicted samples. Use `zero_division` parameter to control this behavior.
_warn_prf(average, modifier, msg_start, len(result))
'''
import warnings
warnings.filterwarnings("ignore")


# 模型评估
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# 使用和之前相同的评估方法
print('1.The accuracy_score of our model is:',accuracy_score(y_test_class,y_pred_class))
print('2.confusion_matrix:')
print(confusion_matrix(y_test_class,y_pred_class))
print('3.classification_report:')
report = classification_report(y_test_class, y_pred_class)
print(report)
'''
precision表示测试的数据是否都预测准确
recall表示需要查的数据是否都查到了
f1=2*(precision*recall)/(precision+recall)
support表示测试数据中属于各个分类的测试数据各有多少个
由此可观察到，此时测试数据在模型上的准确率达到了100%
'''
