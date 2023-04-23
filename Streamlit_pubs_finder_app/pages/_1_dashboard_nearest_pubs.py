import streamlit as st
import pandas as pd
import os
import folium
from sklearn import neighbors
from streamlit_folium import folium_static

main_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_file_name = os.path.join(main_folder_path,'resources','data','pub_df.csv')

pubs_df = pd.read_csv(csv_file_name)


st.title('Find the Nearest Pubs:beers:')
lat = st.number_input('Enter your latitude here: ', value=40)
lon = st.number_input('Enter your longitude here: ', value=50)


dist = neighbors.DistanceMetric.get_metric('haversine')
distances = dist.pairwise(pubs_df[['latitude', 'longitude']], [[lat, lon]])
pubs_df['Distance'] = distances[:, 0]
nearest_pubs = pubs_df.nsmallest(5, 'Distance')

st.title('Nearest Pubs:beers:')
m = folium.Map(location=[lat, lon], zoom_start=5)

nearest_pubs.reset_index(inplace=True)

for index, pub in nearest_pubs.iterrows():
    icon = folium.Icon(color='blue', icon_color='white',icon='beer', prefix='fa')
    folium.Marker(location=[pub['latitude'], pub['longitude']], 
                popup=pub['name'], icon=icon).add_to(m)
folium_static(m)


st.title('Nearest Pubs:beers: List')

st.write('### ' 'S.No', 'Pub Name','Address','City', 'Distance')
for index, nearest_pub in nearest_pubs.iterrows():
    st.write(index, nearest_pub['name'], nearest_pub['address'],nearest_pub['city'], nearest_pub['Distance'])