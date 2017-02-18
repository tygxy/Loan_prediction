# Loan_prediction

## 目录
- 1. 数据导入
- 2. 认识数据
 -  数据预览
 -  数据分析
- 3. 数据清洗  
 - 缺省值处理 
 - 类目特征因子化
 - 标准化
- 4. 特征工程
- 5. 模型训练
- 6. 预测与评估

## 1. 数据导入
```python
# coding:utf-8
# 库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation

sns.set(style='whitegrid',color_codes=True)

# 导入数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 将标识Y/N改为1/0
train.loc[train['Loan_Status'] == 'Y','Loan_Status'] = 1
train.loc[train['Loan_Status'] == 'N','Loan_Status'] = 0
train['Loan_Status'] = train['Loan_Status'].astype('int')
```

## 2. 认识数据
### 2.1 数据预览
- info()
```python
> print train.info()

RangeIndex: 614 entries, 0 to 613
Data columns (total 13 columns):
Loan_ID              614 non-null object
Gender               601 non-null object
Married              611 non-null object
Dependents           599 non-null object
Education            614 non-null object
Self_Employed        582 non-null object
ApplicantIncome      614 non-null int64
CoapplicantIncome    614 non-null float64
LoanAmount           592 non-null float64
Loan_Amount_Term     600 non-null float64
Credit_History       564 non-null float64
Property_Area        614 non-null object
Loan_Status          614 non-null int64
dtypes: float64(4), int64(2), object(7)
memory usage: 62.4+ KB
None
```

  训练集共614个数据，12个特征，1个标识，有缺省值。
  
-  head()
```python
> print train.head()

    Loan_ID Gender Married Dependents     Education Self_Employed  \
0  LP001002   Male      No          0      Graduate            No   
1  LP001003   Male     Yes          1      Graduate            No   
2  LP001005   Male     Yes          0      Graduate           Yes   
3  LP001006   Male     Yes          0  Not Graduate            No   
4  LP001008   Male      No          0      Graduate            No   

   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \
0             5849                0.0         NaN             360.0   
1             4583             1508.0       128.0             360.0   
2             3000                0.0        66.0             360.0   
3             2583             2358.0       120.0             360.0   
4             6000                0.0       141.0             360.0   

   Credit_History Property_Area  Loan_Status  
0             1.0         Urban            1  
1             1.0         Rural            0  
2             1.0         Urban            1  
3             1.0         Urban            1  
4             1.0         Urban            1 
```
 看看前五组数据，有个初步认识。

- describe()
```python
> print train.describe()

       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \
count       614.000000         614.000000  592.000000         600.00000   
mean       5403.459283        1621.245798  146.412162         342.00000   
std        6109.041673        2926.248369   85.587325          65.12041   
min         150.000000           0.000000    9.000000          12.00000   
25%        2877.500000           0.000000  100.000000         360.00000   
50%        3812.500000        1188.500000  128.000000         360.00000   
75%        5795.000000        2297.250000  168.000000         360.00000   
max       81000.000000       41667.000000  700.000000         480.00000   

       Credit_History  Loan_Status  
count      564.000000   614.000000  
mean         0.842199     0.687296  
std          0.364878     0.463973  
min          0.000000     0.000000  
25%          1.000000     0.000000  
50%          1.000000     1.000000  
75%          1.000000     1.000000  
max          1.000000     1.000000  

```

## 2.2 数据分析
### 2.2.1 查看特征的数量，范围等信息
- 输出某个特征的种类和数量比(以特征Gender为例)
```python
print train['Gender'].value_counts(normalize=True)

Male      0.813644
Female    0.186356
Name: Gender, dtype: float64
```

- 用图像表示某个特征的数值(以特征ApplicantIncome为例)
```python
sns.countplot(train['ApplicantIncome'])
plt.show()
```
![](raw/figure_2.png?raw=true)

### 2.2.2 分析特征与标识的关系
- 以特征Married为例
```python
print train['Loan_Status'].groupby(train['Married']).mean()
sns.countplot(train['Married'],hue=train['Loan_Status'])
plt.show()

Married
No     0.629108
Yes    0.716080

# Married为Yes中有71.6%的贷款申请通过，Married为No中有62.9%贷款申请通过，所以看到Married这个特征对贷款申请有影响
```

![](raw/figure_3.png?raw=true)

- 以特征ApplicantIncome为例
```python 
print train['Loan_Status'].groupby(pd.qcut(train['ApplicantIncome'],10)).mean()
sns.countplot(pd.qcut(train['ApplicantIncome'],10),hue=train['Loan_Status'])
plt.show()

ApplicantIncome
[150, 2216.1]       0.661290
(2216.1, 2605.4]    0.721311
(2605.4, 3050.4]    0.704918
(3050.4, 3406.8]    0.709677
(3406.8, 3812.5]    0.639344
(3812.5, 4343.6]    0.737705
(4343.6, 5185.6]    0.612903
(5185.6, 6252.4]    0.721311
(6252.4, 9459.9]    0.688525
(9459.9, 81000]     0.677419

# 把ApplicantIncome分为10段，分别观察不同ApplicantIncome与Loan_Status的关系
```
![](raw/figure_4.png?raw=true)

## 3. 数据清洗
### 3.1 缺省值处理
- 直接填补成最常见的情况
```python
# 由于借款时间最常见的为360个月，所以把空缺值补为360，此外把所有Loan_Amount_Term只分为两类，Short_Term和Long_Term
def Loan_Amount_Term_impute(train,test):
	for i in [train,test]:
		i['Loan_Amount_Term'].fillna('360.0',inplace=True)
		i['Loan_Amount_Term'] = np.where((i['Loan_Amount_Term'])<=180,'Short_Term','Long_Term')
	return train,test
```

- 使用一个全局变量NaN代替
```python
def Credit_History_impute(train,test):
	for i in [train,test]:
		i['Credit_History'].fillna('NaN',inplace=True)
	return train,test
```

- 建模法
```python
# 借用Self_Employed和Education两个属性预测LoanAmount
def LoanAmount_impute(train,test):
	table = train.pivot_table(values='LoanAmount',index='Self_Employed',columns='Education',aggfunc=np.median)
	def fage(x):
		return table.loc[x['Self_Employed'],x['Education']]
	train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage,axis=1),inplace=True)
	test['LoanAmount'].fillna(test[test['LoanAmount'].isnull()].apply(fage,axis=1),inplace=True)
	return train,test
```

### 3.2 类目特征因子化
```python
def dummies(train,test,columns=['Credit_History','Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Property_Area']):
	for column in columns:
		train[column] = train[column].apply(lambda x:str(x))
		test[column] = test[column].apply(lambda x:str(x))
		good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
		train = pd.concat((train,pd.get_dummies(train[column],prefix=column)[good_cols]),axis=1)
		test = pd.concat((test,pd.get_dummies(test[column],prefix=column)[good_cols]),axis=1)
		del train[column]
		del test[column]
	return train,test
```

### 3.3 标准化
```python
def scaler(train,test):
	scaler = preprocessing.StandardScaler()
	train['ApplicantIncome'] = pd.Series(scaler.fit_transform(train['ApplicantIncome'].reshape(-1,1)).reshape(-1), index=train.index)
	test['ApplicantIncome'] = pd.Series(scaler.fit_transform(test['ApplicantIncome'].reshape(-1,1)).reshape(-1), index=test.index)
	train['CoapplicantIncome'] = pd.Series(scaler.fit_transform(train['CoapplicantIncome'].reshape(-1,1)).reshape(-1), index=train.index)
	test['CoapplicantIncome'] = pd.Series(scaler.fit_transform(test['CoapplicantIncome'].reshape(-1,1)).reshape(-1), index=test.index)
	train['LoanAmount'] = pd.Series(scaler.fit_transform(train['LoanAmount'].reshape(-1,1)).reshape(-1), index=train.index)
	test['LoanAmount'] = pd.Series(scaler.fit_transform(test['LoanAmount'].reshape(-1,1)).reshape(-1), index=test.index)
	return train,test
```
```python
# 查看数据清洗后的数据
print train.info()

# 数据齐全，且特征已经因子化
RangeIndex: 614 entries, 0 to 613
Data columns (total 24 columns):
ApplicantIncome                614 non-null float64
CoapplicantIncome              614 non-null float64
LoanAmount                     614 non-null float64
Loan_Status                    614 non-null int64
Credit_History_1.0             614 non-null uint8
Credit_History_0.0             614 non-null uint8
Credit_History_NaN             614 non-null uint8
Gender_Male                    614 non-null uint8
Gender_Female                  614 non-null uint8
Married_No                     614 non-null uint8
Married_Yes                    614 non-null uint8
Dependents_0                   614 non-null uint8
Dependents_1                   614 non-null uint8
Dependents_2                   614 non-null uint8
Dependents_3+                  614 non-null uint8
Education_Graduate             614 non-null uint8
Education_Not Graduate         614 non-null uint8
Self_Employed_No               614 non-null uint8
Self_Employed_Yes              614 non-null uint8
Loan_Amount_Term_Long_Term     614 non-null uint8
Loan_Amount_Term_Short_Term    614 non-null uint8
Property_Area_Urban            614 non-null uint8
Property_Area_Rural            614 non-null uint8
Property_Area_Semiurban        614 non-null uint8
dtypes: float64(3), int64(1), uint8(20)
memory usage: 31.2 KB
None
```
## 4. 特征工程
```python
# 选择需要的特征，并对train,test数据进行必要的处理
train = train.filter(regex='Loan_Status|Gender_.*|Married_.*|Dependents_.*|ApplicantIncome|CoapplicantIncome|Credit_History|Education_.*|Self_Employed_.*|Property_Area_.*|Loan_Amount_Term_.*|LoanAmount')
train_y = train.as_matrix()[:,3]
del train['Loan_Status']
train_X=train.as_matrix()[:,0:]

test = test.filter(regex='Loan_ID|Gender_.*|Married_.*|Dependents_.*|ApplicantIncome|CoapplicantIncome|Credit_History|Education_.*|Self_Employed_.*|Property_Area_.*|Loan_Amount_Term_.*|LoanAmount')
test_X=test.as_matrix()[:,1:]

```

## 5. 训练模型(逻辑回归为例)
- 参数微调
```python
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search

parameters={'C': [0.1,2]}
lg = LogisticRegression()
clf = grid_search.GridSearchCV(lg,parameters)
print clf

GridSearchCV(cv=None, error_score='raise',
       estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False),
       fit_params={}, iid=True, n_jobs=1, param_grid={'C': [0.1, 2]},
       pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)
```

- 训练模型
```python
clf = linear_model.LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
clf.fit(train_X,train_y)
```
## 6. 预测与评估
```python
predictions = clf.predict(test_X)

# 将预测结果写入predictions_LR.csv
predictions_copy=[]
for i in predictions:
	if i ==0:
		predictions_copy.append('N')
	elif i==1:
		predictions_copy.append('Y')

result = pd.DataFrame({'Loan_ID':test['Loan_ID'].as_matrix(),'Loan_Status':predictions_copy})
result.to_csv('predictions_LR.csv',index=False)
```

- 结果


![](raw/figure_5.png?raw=true)
