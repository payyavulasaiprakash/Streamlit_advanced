import pandas as pd
import streamlit as st
from matplotlib import image
import os
import ntpath,pickle
import plotly.express as px
import csv
import sys
 
sys.path.append('../laptop_price_prediction')

import linear_regression_updated 

lr_model,dict_data = linear_regression_updated.lr_model_training()

st.title("Dashboard - Laptop dataset analysis")


file_dir,_=ntpath.split((os.path.abspath(__file__)))
image_file_dir=os.path.join(os.path.dirname(file_dir),'resources')
image_file_path = os.path.join(image_file_dir, "images", "laptop.jpg")
dataset_file_path = os.path.join(image_file_dir, "data", "laptop_details_copy.csv")
# model_path=os.path.join(os.path.dirname(file_dir),'finalized_linear_regression_model.sav')

st.subheader("Sample image of Laptop")

img=image.imread(image_file_path)

st.image(img,width=600)

st.subheader("Sample of dataset")

dataset=pd.read_csv(dataset_file_path)

st.dataframe(dataset.head())

column_names=list(dataset.columns)

print(column_names)

Brands=dataset.Brand.unique()

Storages= dataset.Storage.unique()

os_s= dataset.OS.unique()

RAMs= dataset.RAM.unique()

processors= dataset.processor.unique()

brand_selected = st.selectbox("Please select the Brand of the Laptop : ", Brands)

storage_selected = st.selectbox("Please select the storage of the Laptop : ", Storages)

OS_selected = st.selectbox("Please select the OS of the Laptop : ", os_s)

RAM_selected = st.selectbox("Please select the RAM of the Laptop : ", RAMs)

processor_selected = st.selectbox("Please select the Processor of the Laptop : ", processors)

rating = st.slider("Please provide rating for the Laptop",0.0,5.0)

dict_data_final={'processor':[dict_data['unique_processor'].index(processor_selected)],'RAM':[dict_data['unique_rams'].index(RAM_selected)],'OS':[dict_data['unique_OS'].index(OS_selected)],'Storage':[dict_data['unique_Storage'].index(storage_selected)],'Brand':[dict_data['unique_Brand'].index(brand_selected)]}

final_df=pd.DataFrame(dict_data_final)

output=lr_model.predict(final_df)

st.write("For the above requirements Laptop's Price is :", output)
st.write("Thank You")
