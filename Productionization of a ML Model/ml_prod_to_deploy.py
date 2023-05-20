# importing required values
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, pickle
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

csv_file_name = 'train_v9rqX0R.csv'

def prepare_dataset(csv_file_name=csv_file_name):
    # read the train data
    dataset = pd.read_csv(csv_file_name)

    dataset.Item_Weight.fillna(dataset.Item_Weight.median(),inplace=True)

    dataset.Outlet_Size.fillna(dataset.Outlet_Size.mode()[0],inplace=True)
    print('columns',dataset.columns) 
    OHE = ce.OneHotEncoder(cols=['Item_Fat_Content',
                             'Item_Type',
                             'Outlet_Identifier',
                             'Outlet_Size',
                             'Outlet_Location_Type',
                             'Outlet_Type'],use_cat_names=True)
# encode the categorical variables
    dataset = OHE.fit_transform(dataset)
    dataset_X = dataset.drop(columns=['Item_Identifier','Item_Outlet_Sales'])
    dataset_Y = dataset['Item_Outlet_Sales']
    

    train_x, test_x, train_y, test_y = train_test_split(dataset_X, dataset_Y,test_size=0.25,random_state=0)

    scaler = StandardScaler()
    # fit with the Item_MRP
    scaler.fit(np.array(train_x.Item_MRP).reshape(-1,1))
    # transform the data
    train_x.Item_MRP = scaler.transform(np.array(train_x.Item_MRP).reshape(-1,1))
    test_x.Item_MRP = scaler.transform(np.array(test_x.Item_MRP).reshape(-1,1))

    with open('one_hot_encoding.pkl','wb') as file:
        pickle.dump(OHE,file)

    with open('StandardScaler.pkl','wb') as file:
        pickle.dump(scaler,file)
    return train_x, test_x, train_y, test_y 


def fit(train_x,train_y,estimator):
    estimator = estimator
    estimator.fit(train_x, train_y)
    return estimator

def test(data,estimator,standard_scaler,scaling = False,):
    if scaling:
        data.Item_MRP = standard_scaler.transform(np.array(data.Item_MRP).reshape(-1,1))
        predict_test  = estimator.predict(data)
    else:
        predict_test  = estimator.predict(data)
    return predict_test



train_x, test_x, train_y, test_y  =  prepare_dataset(csv_file_name=csv_file_name)
with open('StandardScaler.pkl','rb') as file:
        standard_scaler = pickle.load(file)
 
model_LR = LinearRegression()
model_LR = fit(train_x,train_y,model_LR)

lr_model_predictions_train = test(train_x,model_LR,standard_scaler, False)
lr_model_predictions_test = test(test_x,model_LR,standard_scaler,False)

# Root Mean Squared Error on train and test date

lr_model_train_rmse = mean_squared_error(train_y, lr_model_predictions_train)**(0.5)
lr_model_test_rmse = mean_squared_error(test_y, lr_model_predictions_test)**(0.5)

print(lr_model_train_rmse, lr_model_test_rmse)

with open('lr_model.pkl','wb') as file:
        pickle.dump(model_LR,file)




