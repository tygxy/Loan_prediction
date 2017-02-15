# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation


sns.set(style='whitegrid',color_codes=True)

# 0.导入数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.loc[train['Loan_Status'] == 'Y','Loan_Status'] = 1
train.loc[train['Loan_Status'] == 'N','Loan_Status'] = 0
train['Loan_Status'] = train['Loan_Status'].astype('int')


# 1.数据初览
# print train.info()
# print train.head()
# print train.describe()

# 2.数据初探

## 2.1 Loan_Status
# print train['Loan_Status'].value_counts(normalize=True)
# sns.countplot(train['Loan_Status'])
# plt.show()

## 2.2 Gender(关系不大)
# print train['Loan_Status'].groupby(train['Gender']).mean()
# sns.countplot(train['Gender'],hue=train['Loan_Status'])
# plt.show()


## 2.3 Married(弱关系)
# print train['Loan_Status'].groupby(train['Married']).mean()
# sns.countplot(train['Married'],hue=train['Loan_Status'])
# plt.show()


## 2.3 Dependents(弱关系)
# print train['Loan_Status'].groupby(train['Dependents']).mean()
# sns.countplot(train['Dependents'],hue=train['Loan_Status'])
# plt.show()

## 2.4 Education(弱关系)
# print train['Loan_Status'].groupby(train['Education']).mean()
# sns.countplot(train['Education'],hue=train['Loan_Status'])
# plt.show()

## 2.5 Self_Employed(关系不大)
# print train['Loan_Status'].groupby(train['Self_Employed']).mean()
# sns.countplot(train['Self_Employed'],hue=train['Loan_Status'])
# plt.show()

## 2.6 ApplicantIncome(有关系)

# sns.distplot(a=train['ApplicantIncome'])
# sns.plt.show()

# print train['Loan_Status'].groupby(pd.qcut(train['ApplicantIncome'],50)).mean()
# sns.countplot(pd.qcut(train['ApplicantIncome'],10),hue=train['Loan_Status'])
# plt.show()

## 2.7 CoapplicantIncome


## 2.8 LoanAmount
# sns.distplot(a=train['ApplicantIncome'])
# sns.plt.show()

# print train['Loan_Status'].groupby(pd.qcut(train['LoanAmount'],5)).mean()
# sns.countplot(pd.qcut(train['LoanAmount'],5),hue=train['Loan_Status'])
# plt.show()

## 2.9 Loan_Amount_Term(短期容易，长期难)
# print train['Loan_Status'].groupby(pd.qcut(train['Loan_Amount_Term'],2)).mean()
# sns.countplot(pd.qcut(train['Loan_Amount_Term'],2),hue=train['Loan_Status'])
# plt.show()


## 2.10 Credit_History(强)
# print train['Loan_Status'].groupby(train['Credit_History']).mean()
# sns.countplot(train['Credit_History'],hue=train['Loan_Status'])
# plt.show()

## 2.11 Property_Area
# print train['Loan_Status'].groupby(train['Property_Area']).mean()
# sns.countplot(train['Property_Area'],hue=train['Loan_Status'])
# plt.show()


# 3. 数据预处理

## 3.1 Self_Employed
def Self_Employed_impute(train,test):
	for i in [train,test]:
		i['Self_Employed'].fillna('No',inplace=True)
	return train,test


# 3.2 LoanAmount
def LoanAmount_impute(train,test):
	table = train.pivot_table(values='LoanAmount',index='Self_Employed',columns='Education',aggfunc=np.median)
	def fage(x):
		return table.loc[x['Self_Employed'],x['Education']]
	train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage,axis=1),inplace=True)
	test['LoanAmount'].fillna(test[test['LoanAmount'].isnull()].apply(fage,axis=1),inplace=True)
	return train,test

# 3.3 Gender
def Gender_impute(train,test):
	for i in [train,test]:
		i['Gender'].fillna('Male',inplace=True)
	return train,test

# 3.4 Married
def Married_impute(train,test):
	for i in [train,test]:
		i['Married'].fillna('Yes',inplace=True)
	return train,test	

# 3.4 Dependents
def Dependents_impute(train,test):
	for i in [train,test]:
		i['Dependents'].fillna('0',inplace=True)
	return train,test	

# 3.5 Loan_Amount_Term
def Loan_Amount_Term_impute(train,test):
	for i in [train,test]:
		i['Loan_Amount_Term'].fillna('360.0',inplace=True)
		i['Loan_Amount_Term'] = np.where((i['Loan_Amount_Term'])<=180,'Short_Term','Long_Term')
	return train,test

# 3.6 Credit_History
def Credit_History_impute(train,test):
	# table = train.pivot_table(values='Credit_History',index='ApplicantIncome',columns='Education',aggfunc=np.median)
	# def fage(x):
	# 	return table.loc[x['ApplicantIncome'],x['Education']]
	# train['Credit_History'].fillna(train[train['Credit_History'].isnull()].apply(fage,axis=1),inplace=True)
	# test['Credit_History'].fillna(test[test['Credit_History'].isnull()].apply(fage,axis=1),inplace=True)
	# return train,test
	for i in [train,test]:
		i['Credit_History'].fillna('NaN',inplace=True)
	return train,test

# 3.7 dummies
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

def drop(train):
	del train['Loan_ID']
	return train

def scaler(train,test):
	scaler = preprocessing.StandardScaler()
	ApplicantIncome_scale_param = scaler.fit(train['ApplicantIncome'])
	train['ApplicantIncome'] = scaler.fit_transform(train['ApplicantIncome'],ApplicantIncome_scale_param)
	test['ApplicantIncome'] = scaler.fit_transform(test['ApplicantIncome'],ApplicantIncome_scale_param)

	CoapplicantIncome_scale_param = scaler.fit(train['CoapplicantIncome'])
	train['CoapplicantIncome'] = scaler.fit_transform(train['CoapplicantIncome'],CoapplicantIncome_scale_param)
	test['CoapplicantIncome'] = scaler.fit_transform(test['CoapplicantIncome'],CoapplicantIncome_scale_param)

	LoanAmount_scale_param = scaler.fit(train['LoanAmount'])
	train['LoanAmount'] = scaler.fit_transform(train['LoanAmount'],LoanAmount_scale_param)
	test['LoanAmount'] = scaler.fit_transform(test['LoanAmount'],LoanAmount_scale_param)	
	return train,test

train,test = Self_Employed_impute(train,test)
train,test = LoanAmount_impute(train,test)
train,test = Gender_impute(train,test)
train,test = Married_impute(train,test)
train,test = Dependents_impute(train,test)
train,test = Loan_Amount_Term_impute(train,test)
train,test = Credit_History_impute(train,test)
train,test = dummies(train,test)
# train,test = scaler(train,test)
train = drop(train)

# 4.训练模型

# 4.1 数据处理

# 4.1.1整体数据处理
train = train.filter(regex='Loan_Status|Gender_.*|Married_.*|Dependents_.*|ApplicantIncome|CoapplicantIncome|Credit_History|Education_.*|Self_Employed_.*|Property_Area_.*|Loan_Amount_Term_.*|LoanAmount')
train_y = train.as_matrix()[:,3]
del train['Loan_Status']
train_X=train.as_matrix()[:,0:]

test = test.filter(regex='Loan_ID|Gender_.*|Married_.*|Dependents_.*|ApplicantIncome|CoapplicantIncome|Credit_History|Education_.*|Self_Employed_.*|Property_Area_.*|Loan_Amount_Term_.*|LoanAmount')
test_X=test.as_matrix()[:,1:]

# 4.1.2交叉数据处理
# split_train,split_cv = cross_validation.train_test_split(train,test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Loan_Status|Gender_.*|Married_.*|Dependents_.*|ApplicantIncome|CoapplicantIncome|Credit_History|Education_.*|Self_Employed_.*|Property_Area_.*|Loan_Amount_Term_.*|LoanAmount')
# # train_df = split_train.filter(regex='Loan_Status|ApplicantIncome|CoapplicantIncome|Credit_History|LoanAmount')

# train_y=train_df.as_matrix()[:,3]
# del train_df['Loan_Status']
# train_X=train_df.as_matrix()[:,0:]

# cv_df = split_cv.filter(regex='Loan_Status|Gender_.*|Married_.*|Dependents_.*|ApplicantIncome|CoapplicantIncome|Credit_History|Education_.*|Self_Employed_.*|Property_Area_.*|Loan_Amount_Term_.*|LoanAmount')
# # cv_df = split_cv.filter(regex='Loan_Status|ApplicantIncome|CoapplicantIncome|Credit_History|LoanAmount')
# test_y = cv_df.as_matrix()[:,3]
# del cv_df['Loan_Status']
# test_X=cv_df.as_matrix()[:,0:]

## 4.2 逻辑回归
clf = linear_model.LogisticRegression(C=1.0,tol=1e-6)
clf.fit(train_X,train_y)
predictions = clf.predict(test_X)
# print metrics.accuracy_score(predictions,test_y)

## 4.3 RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(criterion='gini',n_estimators=700,min_samples_split=10,
# 							min_samples_leaf=1,max_features='auto',oob_score=True,random_state=1,n_jobs=-1)

# rf.fit(train_X,train_y)
# predictions = rf.predict(test_X)

# print pd.concat((pd.DataFrame(train_df.iloc[:, 0:].columns, columns = ['variable']), 
#            pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
#           axis = 1).sort_values(by='importance', ascending = False)[:5]


## 5 输出预测结果
predictions_copy=[]
for i in predictions:
	if i ==0:
		predictions_copy.append('N')
	elif i==1:
		predictions_copy.append('Y')

result = pd.DataFrame({'Loan_ID':test['Loan_ID'].as_matrix(),'Loan_Status':predictions_copy})
result.to_csv('predictions_LR.csv',index=False)

