import pandas as pd
import streamlit as st
from matplotlib import image
import os
import ntpath,pickle
import plotly.express as px
import csv
import sys
import numpy as np
 
st.title("Dashboard - Big Mart Sales")


file_dir,_=ntpath.split((os.path.abspath(__file__)))
image_file_dir=os.path.join(os.path.dirname(file_dir),'resources')
image_file_path = os.path.join(image_file_dir, "images", "big-mart-sales.jpg")
dataset_file_path = os.path.join(image_file_dir, "data", "train_v9rqX0R.csv")

column_names = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']


with open(os.path.join(os.path.dirname(file_dir),'StandardScaler.pkl'),'rb') as file:
    standard_scaler = pickle.load(file)

with open(os.path.join(os.path.dirname(file_dir),'lr_model.pkl'),'rb') as file:
    lr_model = pickle.load(file)

with open(os.path.join(os.path.dirname(file_dir),'one_hot_encoding.pkl'),'rb') as file:
    one_hot_encoder = pickle.load(file)

img=image.imread(image_file_path)

st.image(img,width=600)

st.write("-------------------------------------------------------source: google----------------------------------------------------")

st.subheader("Sample of Big Mart Sales dataset")

dataset=pd.read_csv(dataset_file_path)

st.dataframe(dataset.head())

column_names=list(dataset.columns)

# print(column_names)

Item_Type=dataset.Item_Type.unique()

Outlet_Identifier= dataset.Outlet_Identifier.unique()

Item_Fat_Content= dataset.Item_Fat_Content.unique()

Outlet_Size= dataset.Outlet_Size.unique()

Outlet_Location_Type= dataset.Outlet_Location_Type.unique()

Outlet_Type= dataset.Outlet_Type.unique()

Item_Type_selected = st.selectbox("Please select the Item_Type : ", Item_Type)

Outlet_Identifier_selected = st.selectbox("Please select the Outlet_Identifier : ", Outlet_Identifier)

Item_Fat_Content_selected = st.selectbox("Please select the Item_Fat_Content : ", Item_Fat_Content)

Outlet_Size_selected = st.selectbox("Please select the Outlet_Size : ", Outlet_Size)

Outlet_Location_Type_selected = st.selectbox("Please select the Outlet_Location_Type : ", Outlet_Location_Type)

Outlet_Type_selected = st.selectbox("Please select the Outlet_Location_Type : ", Outlet_Type)

Item_Weight = number = st.number_input('Insert an Item_Weight')

Item_MRP = st.number_input('Insert an Item_MRP')

Item_Visibility = st.number_input('Insert an Item_Visibility')

Outlet_Establishment_Year = st.number_input('Insert an Outlet_Establishment_Year')

dict_data = {'Item_Identifier':[1],'Item_Weight':[Item_Weight], 'Item_Fat_Content':[Item_Fat_Content_selected], 'Item_Visibility':[Item_Visibility],
       'Item_Type':[Item_Type_selected], 'Item_MRP':[Item_MRP], 'Outlet_Identifier':[Outlet_Identifier_selected],
       'Outlet_Establishment_Year':[Outlet_Establishment_Year], 'Outlet_Size':[Outlet_Size_selected], 'Outlet_Location_Type':[Outlet_Location_Type_selected],
       'Outlet_Type':[Outlet_Type_selected],'Item_Outlet_Sales':[1]}

df = pd.DataFrame(dict_data)

# print(df.head())

ohe_out = one_hot_encoder.transform(df)

# print('type',type(ohe_out.Item_MRP))

ohe_out.Item_MRP = standard_scaler.transform(np.array(ohe_out.Item_MRP).reshape(-1,1))

ohe_out = ohe_out.drop(columns=['Item_Identifier','Item_Outlet_Sales'])

output=lr_model.predict(ohe_out)

st.write("For the above requirements Item_Outlet_Sales is :", output[0])

st.write("Thank You for visiting")
