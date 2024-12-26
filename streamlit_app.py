import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import ParameterGrid

# Darts kütüphanesi işlevleri
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel

import io

st.title('🎈 Hava Kirliliği Tahmini Uygulaması')

st.info('Bu uygulama derin öğrenme modeli ile tahmin yapar!')



# Veri yükleme ve işleme
uploaded_file = st.file_uploader("Lütfen hava kalitesi verisini yükleyin (CSV formatında)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col="ds", parse_dates=True)
    data = data.asfreq("d")

    # Veriyi görselleştirme
    st.subheader("Günlük PM10 Değeri Grafiği")
    fig, ax = plt.subplots(figsize=(10, 6))
    data['y'].plot(title='Günlük PM10 Değeri', ax=ax)
    st.pyplot(fig)

    # Otokorelasyon ve Parsiyel Otokorelasyon Grafikleri
    st.subheader("Otokorelasyon Grafiği")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(data['y'], lags=100, ax=ax)
    st.pyplot(fig)

    st.subheader("Parsiyel Otokorelasyon Grafiği")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(data['y'], lags=100, ax=ax)
    st.pyplot(fig)

else:
    st.warning("Lütfen bir CSV dosyası yükleyin.")
