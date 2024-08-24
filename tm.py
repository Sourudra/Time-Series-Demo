import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit app
st.title('Time Series Analysis App')

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file, parse_dates=['Month'], index_col='Month')
    
    # Check if the data contains the correct columns
    if 'Passengers' not in df.columns:
        st.error("CSV file must contain a 'Passengers' column.")
    else:
        st.write("Data successfully loaded:")
        st.write(df.head())

        # Plot the time series data
        st.subheader('Time Series Plot')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Passengers'], marker='o', linestyle='-')
        ax.set_title('Monthly Airline Passenger Numbers')
        ax.set_xlabel('Month')
        ax.set_ylabel('Passengers')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)

        # Perform time series decomposition
        st.subheader('Time Series Decomposition')
        try:
            decomposition = seasonal_decompose(df['Passengers'], model='additive')
            fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
            
            ax[0].plot(decomposition.observed)
            ax[0].set_title('Observed')
            
            ax[1].plot(decomposition.trend)
            ax[1].set_title('Trend')
            
            ax[2].plot(decomposition.seasonal)
            ax[2].set_title('Seasonal')
            
            ax[3].plot(decomposition.resid)
            ax[3].set_title('Residual')
            
            plt.xlabel('Month')
            st.pyplot(fig)
            
            # Analyze components
            st.write("Analysis of the time series:")
            
            # Trend
            if decomposition.trend.dropna().mean() != 0:
                st.write("- The time series has a trend component.")
            else:
                st.write("- No significant trend detected.")
            
            # Seasonality
            if decomposition.seasonal.dropna().mean() != 0:
                st.write("- The time series has a seasonal component.")
            else:
                st.write("- No significant seasonality detected.")
            
            # Irregularity
            if decomposition.resid.dropna().std() > 10:  # Arbitrary threshold for irregularity
                st.write("- The time series has irregular components.")
            else:
                st.write("- No significant irregularity detected.")
            
            # Stationarity
            result = adfuller(df['Passengers'].dropna())
            if result[1] < 0.05:
                st.write("- The time series is stationary.")
            else:
                st.write("- The time series is not stationary.")
            
            # Autocorrelation and Partial Autocorrelation
            lags = min(len(df) - 1, 40)
            acf_values = acf(df['Passengers'].dropna(), nlags=lags)
            pacf_values = pacf(df['Passengers'].dropna(), nlags=lags)
            
            fig, ax = plt.subplots(2, 1, figsize=(12, 10))
            ax[0].stem(range(len(acf_values)), acf_values)
            ax[0].set_title('Autocorrelation Function (ACF)')
            ax[1].stem(range(len(pacf_values)), pacf_values)
            ax[1].set_title('Partial Autocorrelation Function (PACF)')
            plt.xlabel('Lag')
            st.pyplot(fig)
            
            # Mean Reversion
            mean_reversion = df['Passengers'].rolling(window=12).mean().dropna()
            if mean_reversion.std() > 10:  # Arbitrary threshold for mean reversion
                st.write("- The time series exhibits mean reversion.")
            else:
                st.write("- No significant mean reversion detected.")
            
            # Volatility
            df['Returns'] = df['Passengers'].pct_change().dropna()
            volatility = df['Returns'].std()
            st.write(f"- The volatility of the time series is {volatility:.2f}.")
            
        except Exception as e:
            st.error(f"Error during analysis: {e}")

    # Info Button
    if st.button('Want to know more about Time Series'):
        st.subheader('About Time Series Analysis')
        st.write("""
        **What is Time Series Analysis?**
        
        Time series analysis involves analyzing data points collected or recorded at specific time intervals. It is used to understand the underlying patterns in data that are indexed in time order.

        **Why Use Time Series Analysis?**

        Time series analysis is used to:
        - Forecast future values based on historical data.
        - Identify trends and seasonal patterns.
        - Understand the impact of different variables on the time series.
        - Detect anomalies and irregularities in the data.
        
        **Main Characteristics of Time Series Data:**

        1. **Trend:** The long-term movement or direction in the data over time.
        2. **Seasonality:** Regular, repeating patterns or cycles in the data at consistent intervals.
        3. **Cyclic Patterns:** Fluctuations that occur in cycles but not necessarily at fixed intervals, often influenced by economic or business conditions.
        4. **Irregularity (Noise):** Random variations in the data that cannot be attributed to trend, seasonality, or cyclic patterns.
        5. **Stationarity:** When a time series’ statistical properties like mean and variance are constant over time.
        6. **Autocorrelation:** The correlation of a time series with its own lagged values, measuring how values are related to past values.
        7. **Mean Reversion:** The tendency of a time series to revert to its long-term mean over time.
        8. **Volatility:** The degree of variation or dispersion in the data, indicating stability or fluctuations.

        **Partial Autocorrelation Function (PACF):**
        
        The Partial Autocorrelation Function (PACF) measures the correlation between a time series and its lagged values while controlling for the values of the time series at intermediate lags. It helps identify the direct effect of past observations on the current value and is useful in determining the order of autoregressive models.
        """)

else:
    st.info('Please upload a CSV file to start.')
