import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


st.title("Tesla Stock Price Prediction")

st.write(
    "Stock price prediction using historical Tesla stock data"
)


file = st.file_uploader(
    "Upload TSLA.csv",
    type=["csv"]
)


if file:

    df = pd.read_csv(file)

    st.subheader("Dataset")
    st.write(df.head())


    df["Date"] = pd.to_datetime(df["Date"])

    df=df.sort_values("Date")


    st.subheader(
        "Tesla Closing Price Trend"
    )


    fig, ax = plt.subplots()

    ax.plot(
        df["Date"],
        df["Close"]
    )

    st.pyplot(fig)


    data=df[["Close"]]


    scaler=MinMaxScaler()

    scaled=scaler.fit_transform(data)


    X=[]
    y=[]


    for i in range(60,len(scaled)):

        X.append(
            scaled[i-60:i,0]
        )

        y.append(
            scaled[i,0]
        )


    X=np.array(X)

    y=np.array(y)



    model=LinearRegression()

    model.fit(X,y)



    last_60=scaled[-60:].reshape(1,-1)


    predictions=[]


    current=last_60.copy()


    for i in range(10):

        pred=model.predict(current)

        predictions.append(pred[0])

        current=np.append(
            current[:,1:],
            [[pred[0]]],
            axis=1
        )


    result=scaler.inverse_transform(

        np.array(predictions).reshape(-1,1)

    )


    st.subheader(
        "Future Prediction"
    )


    st.write(
        "Next Day:",
        result[0][0]
    )


    st.write(
        "Next 5 Days:"
    )

    st.write(
        result[:5]
    )


    st.write(
        "Next 10 Days:"
    )

    st.write(
        result
    )
