import streamlit as st
import matplotlib.pyplot as plt
from ml_model import train_and_predict

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide"
)

# ------------------ Title ------------------
st.title("📈 Stock Price Prediction App")
st.write(
    "This app predicts the **next-day stock price** using Machine Learning "
    "based on historical data."
)

# ------------------ Stock Selection ------------------
st.subheader("Select Stock")

stock = st.selectbox(
    "Available Stocks",
    [
        "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN",
        "META", "NFLX", "NVDA",
        "RELIANCE", "TCS", "INFY"
    ]
)

st.write("📂 Data source: Local CSV files (stable & demo-friendly)")

# ------------------ Predict Button ------------------
if st.button("Predict Price"):
    try:
        with st.spinner("Training model and making prediction..."):
            df = train_and_predict(stock)

        st.success("Prediction completed successfully!")

        # ------------------ Plot ------------------
        st.subheader("📊 Actual vs Predicted Price")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Close"], label="Actual Price")
        ax.plot(df.index, df["Predicted"], label="Predicted Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # ------------------ Latest Prediction ------------------
        st.subheader("📌 Latest Prediction")
        st.write(
            f"**Predicted next-day price for {stock}:** "
            f"{df['Predicted'].iloc[-1]:.2f}"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ------------------ Footer ------------------
st.markdown("---")
st.caption(
    "⚠️ This project is for educational purposes only. "
    "It does NOT provide financial advice."
)
