import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
import os, ntpath


def lr_model_training():
    file_dir,_=ntpath.split(os.path.abspath(__file__))
    image_file_dir=os.path.join(file_dir,'resources')
    image_file_path = os.path.join(image_file_dir, "images", "laptop.jpg")
    dataset_file_path = os.path.join(image_file_dir, "data", "laptop_details_copy.csv")

    df= pd.read_csv(dataset_file_path)

    features=df.drop(columns=['Feature','Product','MRP','processor_gen'])

    label=df['MRP']
    features['Rating'].fillna(features['Rating'].mean(),inplace=True)
    features=features.drop(columns=['Rating'])

    unique_rams=features['RAM'].unique().tolist()
    unique_processor=features['processor'].unique().tolist()
    unique_OS=features['OS'].unique().tolist()
    unique_Brand=features['Brand'].unique().tolist()
    unique_Storage=features['Storage'].unique().tolist()

    features['RAM']=features['RAM'].apply(lambda x : unique_rams.index(x))
    features['processor']=features['processor'].apply(lambda x : unique_processor.index(x))
    features['OS']=features['OS'].apply(lambda x : unique_OS.index(x))
    features['Brand']=features['Brand'].apply(lambda x : unique_Brand.index(x))
    features['Storage']=features['Storage'].apply(lambda x : unique_Storage.index(x))
    print("features.columns",features.columns)
    x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.25,random_state=42)

    lr_model=LinearRegression()
    lr_model.fit(x_train,y_train)
    x_train_pred=lr_model.predict(x_train)
    x_test_pred=lr_model.predict(x_test)

    print(r2_score(x_train_pred,y_train))
    print(r2_score(x_test_pred,y_test))
    
    filename = 'finalized_linear_regression_model.sav'
    pickle.dump(lr_model, open(filename, 'wb'))

    dict_data={"unique_Storage":unique_Storage,"unique_Brand":unique_Brand,'unique_OS':unique_OS,"unique_processor":unique_processor,'unique_rams':unique_rams}
    return lr_model,dict_data

lr_model_training()
