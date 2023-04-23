import streamlit as st
import pandas as pd
import os
import folium
from matplotlib import image
from geopy.geocoders import Nominatim
from sklearn.neighbors import DistanceMetric
from streamlit_folium import folium_static
from PIL import Image

main_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_file_name = os.path.join(main_folder_path,'resources','data','pub_df.csv')

pubs_df = pd.read_csv(csv_file_name)

st.title(":red[Pub Locations] :beers:")

unique_postal_codes = pubs_df['postcode'].unique()
unique_local_authority = pubs_df['local_authority'].unique()

postal_code_selected = st.selectbox("Enter your Postal Code: ", unique_postal_codes)
local_authority_selected = st.selectbox("Enter your Local Authority: ", unique_local_authority)

geolocator = Nominatim(user_agent='my_application')
location_data = geolocator.geocode(f'{postal_code_selected} {local_authority_selected}', exactly_one=False)[0]
lat, lon = location_data.latitude, location_data.longitude

m = folium.Map(location=[lat, lon], zoom_start=13)

filtered_pubs_df = pubs_df[(pubs_df['postcode'] == location_data.raw.get('postcode', '')) |
                        (pubs_df['city'] == location_data.raw.get('city', '')) |
                        (pubs_df['local_authority'] == local_authority_selected)]

for i, pub in filtered_pubs_df.iterrows():
    folium.Marker(location=[pub['latitude'], pub['longitude']], 
                popup=pub['name']).add_to(m)

st.markdown('### Location of pubs:beers: : ')
st.write(f'Number of pubs @ {postal_code_selected} {local_authority_selected}: {len(filtered_pubs_df)}')
folium_static(m) 

st.markdown('### Pubs:beers: List: ')

for index,pub_name in enumerate(filtered_pubs_df['name'].values):
    st.write(index, pub_name)  