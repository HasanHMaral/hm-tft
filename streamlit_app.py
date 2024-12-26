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

# TimeSeries Nesnesi Oluşturma
    st.subheader("TimeSeries Nesnesi")
    try:
        # TimeSeries nesnesini oluşturma
        zaman_serisi = TimeSeries.from_dataframe(data, value_cols="y")
        st.write("TimeSeries Nesnesi Başarıyla Oluşturuldu!")
        st.write(zaman_serisi)
    except Exception as e:
        st.error(f"TimeSeries nesnesi oluşturulurken bir hata oluştu: {str(e)}")

# Encoder Fonksiyonları
def yil_kodla(idx):
    """Yıl bilgilerini kodlayan özel bir fonksiyon."""
    return (idx.year - 2000) / 50

# Ekleyicilerin Tanımlanması
ekleyiciler = {
    'cyclic': {'future': ['day', 'dayofweek', 'week', 'month']},
    'datetime_attribute': {'future': ['day', 'dayofweek', 'week', 'month']},
    'position': {'past': ['relative'], 'future': ['relative']},
    'custom': {'past': [yil_kodla], 'future': [yil_kodla]},
    'transformer': Scaler(),
    'tz': 'CET'
}

 # Geçmiş bağımsız değişkenlerin seçilmesi
    st.subheader("Geçmiş Bağımsız Değişkenler")
    if data.shape[1] > 2:  # Eğer bağımsız değişkenler varsa
        X_gecmis = data.iloc[:, 2:]  # İlk iki sütun hariç diğer sütunları al
        st.write("Seçilen Bağımsız Değişkenler:")
        st.write(X_gecmis.head())

        try:
            # TimeSeries nesnesi oluşturma
            gecmis_bagimsiz = TimeSeries.from_dataframe(X_gecmis)
            st.write("Geçmiş Bağımsız Değişkenlerin TimeSeries Nesnesi:")
            st.write(gecmis_bagimsiz)
        except Exception as e:
            st.error(f"TimeSeries nesnesi oluşturulurken bir hata oluştu: {str(e)}")
    else:
        st.error("Veride bağımsız değişken sütunları bulunamadı.")
