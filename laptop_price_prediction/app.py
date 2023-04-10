import streamlit as st
import pandas as pd
import webbrowser

st.title("My first :blue[steamlit basic app] :sunglasses:")

st.header("Hello, Welcome to my web application.")

linkdin_link = 'https://www.linkedin.com/in/saiprakash-payyavula'

github_link='https://github.com/payyavulasaiprakash'

st.write("Please feel free to connect with me, by any of the below with a single click")

Linkedin=st.button('Linkedin')
Github=st.button('Github')

if Linkedin:
    webbrowser.open_new_tab(linkdin_link)
elif Github:   
    webbrowser.open_new_tab(github_link)


rating = st.slider("Please rate how well you enjoyed this page and its features",0,10)

st.write("Thanks for the rating, you have given", rating, "out of 10")