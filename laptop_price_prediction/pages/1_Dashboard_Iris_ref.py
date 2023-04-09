import pandas as pd
import streamlit as st
from matplotlib import image
import os
import ntpath
import plotly.express as px

st.title("Dashboard - Iris dataset analysis")


file_dir,_=ntpath.split((os.path.abspath(__file__)))
image_file_dir=os.path.join(os.path.dirname(file_dir),'resources')
image_file_path = os.path.join(image_file_dir, "images", "iris.jpg")
dataset_file_path = os.path.join(image_file_dir, "data", "iris.csv")

st.subheader("Sample image of iris")

img=image.imread(image_file_path)

st.image(img,width=600)

st.subheader("Sample of dataset")

dataset=pd.read_csv(dataset_file_path)

st.dataframe(dataset.head())

features=list(dataset.columns)

dummy=features.pop()

Species=dataset['Species'].unique()

Species_selected = st.selectbox("Please select the species : ",Species)

features_selected = st.selectbox("Please select the feature : ",features)

st.write('You have selected ', Species_selected, f"species and {features_selected} feature:")

col1, col2 = st.columns(2)

fig_1 = px.histogram(dataset[dataset['Species'] == Species_selected], x=features_selected)
col1.plotly_chart(fig_1, use_container_width=True)

fig_2 = px.box(dataset[dataset['Species'] == Species_selected], y=features_selected)
col2.plotly_chart(fig_2, use_container_width=True)
