import streamlit as st
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import requests
from streamlit_lottie import st_lottie

header =st.container()
dataset=st.container()
model_training=st.container()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_atgvstog.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high # canvas
    height=None,
    width=None,
    key=None,
)

@st.cache
def get_data():
    taxi_data=pd.read_csv(filename)
    return get_data

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")

# Load Animation
animation_symbol = "❄"

st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """,
    unsafe_allow_html=True,
)

with header:
    st.title("Welcome to my awesome data science project")
    st.text('In this project I look into the transction of taxi in Ney York..... ')

with dataset:
    st.header("NYC taxi dataset")
    taxi_data=pd.read_csv('data/Taxi_data.csv')
    st.subheader("Pick-up Location ID distribution on the NYC dataset")
    pulocation_dict =pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dict)



with model_training:
    st.header("Time to train the model")
    st.text("Here you get to choose the hyperparameters of the model and see how the performance changes!")

    sel_col, disp_col=st.columns(2)

    max_depth=sel_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox("How many trees should there be?", options=[100,200,300,'No Limit'], index=0)

    sel_col.text("Here is a list of features in my data:")
    sel_col.write(taxi_data.columns)

    input_features =sel_col.text_input("Which features should be used as the input features?", "PULocationID")

    if n_estimators =="No Limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)



    X =taxi_data[[input_features]]
    y = taxi_data[['trip_distance']]

    regr.fit(X,y)
    prediction=regr.predict(y)

    disp_col.subheader("Mean absolute error of the model is:")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean squared error of the model is:")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("R squared score of the model is:")
    disp_col.write(r2_score(y, prediction))
