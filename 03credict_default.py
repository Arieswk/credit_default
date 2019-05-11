# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
#数据加载
data = pd.read_csv('UCI_Credit_Card.csv')
#数据探索
print(data.shape)#查看数据集大小
print(data.describe())#数据集概览
#查看下一个月违约率的情况
next_month = data['default.payment.next.month'].value_counts()
print(next_month)
df = pd.DataFrame({'default.payment.next.month':next_month.index,'values':next_month.values})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(6,6))
plt.title('信用卡违约率客服\n (违约：1，守约：0)')
sns.set_color_codes('pastel')
sns.barplot(x='default.payment.next.month',y = 'values',data=df)
locs,labels = plt.xticks()
plt.show()
#特征选择，去掉id字段、最后一个结果字段即可
data.drop(['ID'],inplace=True,axis=1)#ID没用
target = data['default.payment.next.month'].values
columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns].values
#30%作为测试集，其余作为训练集
train_x,test_x,train_y,test_y = train_test_split(features,target,test_size=0.3,stratify=target,random_state=1)
#构造各种分类器
classifier = [
    SVC(random_state=1,kernel='rbf'),#高斯核函数
    DecisionTreeClassifier(random_state=1,criterion='gini'),
    RandomForestClassifier(random_state=1,criterion='gini'),
    KNeighborsClassifier(metric='minkowski'),
    AdaBoostClassifier(random_state=1),
]
#分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier',
    'adaBoostclassifier',
]
#分类器参数
classifier_param_grid = [
    {'svc__C':[1],'svc__gamma':[0.01]},
    {'decisiontreeclassifier__max_depth':[6,9,11]},
    {'randomforestclassifier__n_estimators':[3,5,6]},
    {'kneighborsclassifier__n_neighbors':[4,6,8]},
    {'adaBoostclassifier__n_estimators':[10,50,100]},
]
#对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,param_grid,score='accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline,param_grid=param_grid,scoring=score)
    search = gridsearch.fit(train_x,train_y)
    print('GridSearch最优参数：',search.best_params_)
    print('GridSearch最优分数：%.4lf'%search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print('准确率%.4lf'%accuracy_score(test_y,predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] =accuracy_score(test_y,predict_y)
    return response

for model,model_name,model_param_grid in zip(classifier,classifier_names,classifier_param_grid):
    pipeline = Pipeline(
        [
            ('scaler',StandardScaler()),
            (model_name,model)
        ]
    )
    res = GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,model_param_grid,score='accuracy')