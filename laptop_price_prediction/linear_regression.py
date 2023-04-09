import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pickle


df= pd.read_csv("resources/data/laptop_details_copy.csv")

print(df.info())

features=df.drop(columns=['Feature','Product','MRP','processor_gen'])

label=df['MRP']
# print(df.describe())

features['Rating'].fillna(features['Rating'].mean(),inplace=True)
print(features.columns)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
features[['OS','Storage','RAM']] = imputer.fit_transform(features[['OS','Storage','RAM']])

print(features.info())

features=pd.get_dummies(features, columns=['OS','Storage','RAM','Brand','processor'],drop_first=True)

print(features.shape)
print(features.columns)
x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.30,random_state=32)

# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
x_train_pred=lr_model.predict(x_train)
x_test_pred=lr_model.predict(x_test)
# print(accuracy_score(x_train_pred,y_train))
# print(accuracy_score(x_test_pred,y_test))
print(r2_score(x_train_pred,y_train))
print(r2_score(x_test_pred,y_test))

print(x_test_pred)

filename = 'finalized_linear_regression_model.sav'
pickle.dump(lr_model, open(filename, 'wb'))


df=pd.DataFrame(columns=features.columns)
df.to_csv('sample.csv',index=False)