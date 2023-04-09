import pandas as pd
import streamlit as st
from matplotlib import image
import os
import ntpath
import plotly.express as px
import matplotlib.pyplot as plt

st.title("Dashboard - titanic dataset analysis")


file_dir,_=ntpath.split((os.path.abspath(__file__)))
image_file_dir=os.path.join(os.path.dirname(file_dir),'resources')
image_file_path = os.path.join(image_file_dir, "images", "titanic.jpg")
dataset_file_path = os.path.join(image_file_dir, "data", "titanic.csv")

st.subheader("Sample image of titanic")

img=image.imread(image_file_path)

st.image(img,width=600)

st.subheader("Sample of dataset")

dataset=pd.read_csv(dataset_file_path)

st.dataframe(dataset.head())

features=list(dataset.columns)

dummy=features.remove('survived')

survived=dataset['survived'].unique()

survival = st.selectbox("Please select the whether the person survived or not : ",survived)

first_feature_selected = st.selectbox("Please select the feature 1 : ",features)

second_feature_selected = st.selectbox("Please select the feature 2 : ",features)

if survival==0:
    st.write('You have selected non survived,', f"{first_feature_selected} feature one and {second_feature_selected} feature two")
elif survival==1:
    st.write('You have selected survived,', f"{first_feature_selected} feature one and {second_feature_selected} feature two")

st.write("Histogram for ",first_feature_selected)
fig_1 = px.histogram(dataset[dataset['survived'] == survival], x=first_feature_selected)
st.plotly_chart(fig_1, use_container_width=True)

st.write("Histogram for ",second_feature_selected)
fig_1 = px.histogram(dataset[dataset['survived'] == survival], x=first_feature_selected)
st.plotly_chart(fig_1, use_container_width=True)


