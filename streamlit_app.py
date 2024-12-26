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



# Dosya yükleme bileşeni
uploaded_file = st.file_uploader("Hava Kalitesi Verisini Yükleyin (CSV formatında)", type="csv")

# Dosya yüklendiyse
if uploaded_file is not None:
    # Veriyi yükleme ve indeksi ayarlama
    data = pd.read_csv(uploaded_file, index_col="ds", parse_dates=True)
    
    # Veri hakkında bilgi StringIO ile yakalama
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Veri hakkında bilgiyi yazdırma
    st.write("Yüklenen Verinin Bilgisi:")
    st.text(info_str)  # Bilgiyi Streamlit'te metin olarak gösterir
    
    # İlk birkaç satırı görüntüleme
    st.write("Verinin İlk 5 Satırı:")
    st.write(data.head())
    
# Veriyi günlük frekansa ayarlama
    data = data.asfreq("d")
# Günlük PM10 Değeri Görselleştirme
    st.subheader("Günlük PM10 Değeri")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['y'])
    ax.set_title("Günlük PM10 Değeri")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("PM10")
    st.pyplot(fig)
# Otokorelasyon Grafiği
    st.subheader("Otokorelasyon Grafiği (ACF)")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(data['y'].dropna(), lags=100, ax=ax)  # Null değerleri kaldırmayı unutmayın
    st.pyplot(fig)
# Parsiyel otokorelasyon grafiği
    st.subheader("Parsiyel Otokorelasyon Grafiği (PACF)")
    if 'y' in data.columns:
        if data['y'].isnull().sum() > 0:
            data['y'] = data['y'].dropna()

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_pacf(data['y'], lags=100, ax=ax, method='ywm')  # PACF grafiği
        st.pyplot(fig)
    else:
        st.error("Veride 'y' sütunu bulunamadı. Lütfen doğru dosyayı yükleyin.")
