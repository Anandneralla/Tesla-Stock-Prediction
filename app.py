import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model


# -----------------------
# Title
# -----------------------

st.title(
    "Tesla Stock Price Prediction using LSTM"
)


st.write(
    "Deep Learning Time Series Forecasting Project"
)



# -----------------------
# Load Model
# -----------------------

model = load_model(
    "tesla_lstm_model.h5"
)



# -----------------------
# Upload Dataset
# -----------------------

uploaded_file = st.file_uploader(
    "Upload Tesla CSV File",
    type=["csv"]
)


if uploaded_file is not None:


    df = pd.read_csv(uploaded_file)


    st.subheader(
        "Dataset Preview"
    )

    st.write(
        df.head()
    )



    df["Date"] = pd.to_datetime(
        df["Date"]
    )


    df.set_index(
        "Date",
        inplace=True
    )


    # Graph


    st.subheader(
        "Tesla Closing Price"
    )


    fig = plt.figure(
        figsize=(12,5)
    )


    plt.plot(
        df["Close"]
    )


    st.pyplot(fig)



    data=df[["Close"]]


    scaler=MinMaxScaler(
        feature_range=(0,1)
    )


    scaled=scaler.fit_transform(
        data
    )


    last_60_days = scaled[-60:]


    future=[]


    temp=list(
        last_60_days.reshape(-1)
    )


    for i in range(10):


        x=np.array(
            temp[-60:]
        )


        x=x.reshape(
            1,60,1
        )


        prediction=model.predict(x)


        temp.append(
            prediction[0][0]
        )


        future.append(
            prediction[0][0]
        )


    result=scaler.inverse_transform(

        np.array(future).reshape(-1,1)

    )


    st.subheader(
        "Future Stock Prediction"
    )


    st.write(
        "Next Day Prediction:"
    )

    st.write(
        result[0][0]
    )


    st.write(
        "Next 5 Days Prediction:"
    )


    st.write(
        result[:5]
    )


    st.write(
        "Next 10 Days Prediction:"
    )


    st.write(
        result
    )
