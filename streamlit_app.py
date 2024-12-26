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

def yukle_ve_isle(dosya_adi):
    """
    Veri yükler, işler ve görselleştirir.

    Args:
        dosya_adi (str): Yüklenecek CSV dosyasının adı.

    Returns:
        pandas.DataFrame: İşlenmiş veri çerçevesi.
    """

    data = pd.read_csv(dosya_adi, index_col='ds', parse_dates=True)
    data = data.asfreq('d')

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

    return data
# TimeSeries Nesnesi ve Bağımsız Değişkenlerin Hazırlanması için Fonksiyonlar
def zaman_serisi_olustur(data):
    """
    Ana zaman serisi nesnesini oluşturur.

    Args:
        data (pandas.DataFrame): İşlenmiş veri çerçevesi.

    Returns:
        TimeSeries: Zaman serisi nesnesi.
    """
    zaman_serisi = TimeSeries.from_dataframe(data, value_cols="y")
    st.write("Zaman Serisi:")
    st.write(zaman_serisi)
    return zaman_serisi

def encoder_ve_ekleyici_olustur():
    """
    Encoder fonksiyonlarını ve ekleyici yapılarını hazırlar.

    Returns:
        dict: Ekleyiciler sözlüğü.
    """
    def yil_kodla(idx):
        return (idx.year - 2000) / 50

    ekleyiciler = {
        'cyclic': {'future': ['day', 'dayofweek', 'week', 'month']},
        'datetime_attribute': {'future': ['day', 'dayofweek', 'week', 'month']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'custom': {'past': [yil_kodla], 'future': [yil_kodla]},
        'transformer': Scaler(),
        'tz': 'CET'
    }
    st.write("Ekleyiciler Tanımlandı.")
    return ekleyiciler

def gecmis_bagimsiz_hazirla(data):
    """
    Geçmiş bağımsız değişkenleri hazırlar.

    Args:
        data (pandas.DataFrame): İşlenmiş veri çerçevesi.

    Returns:
        TimeSeries: Geçmiş bağımsız değişkenlerin zaman serisi nesnesi.
    """
    X_gecmis = data.iloc[:, 2:]
    gecmis_bagimsiz = TimeSeries.from_dataframe(X_gecmis)
    st.write("Geçmiş Bağımsız Değişkenler:")
    st.write(gecmis_bagimsiz)
    return gecmis_bagimsiz

def gelecek_bagimsiz_hazirla(gecmis_bagimsiz, gelecek_veri_dosyasi):
    """
    Gelecek bağımsız değişkenleri hazırlar.

    Args:
        gecmis_bagimsiz (TimeSeries): Geçmiş bağımsız değişkenler.
        gelecek_veri_dosyasi (str): Gelecek bağımsız değişken verilerinin dosya yolu.

    Returns:
        TimeSeries: Gelecek bağımsız değişkenlerin zaman serisi nesnesi.
    """
    gelecek_veri = pd.read_csv(gelecek_veri_dosyasi, index_col='ds', parse_dates=True)
    X_gelecek = gelecek_veri.iloc[:, 1:]
    bagimsiz_tum = pd.concat([gecmis_bagimsiz.pd_dataframe(), X_gelecek])
    gelecek_bagimsiz = TimeSeries.from_dataframe(bagimsiz_tum)
    st.write("Gelecek Bağımsız Değişkenler:")
    st.write(gelecek_bagimsiz)
    return gelecek_bagimsiz
uploaded_file = st.file_uploader("Lütfen hava kalitesi verisini yükleyin (CSV formatında)", type=["csv"])
if uploaded_file is not None:
    data = yukle_ve_isle(uploaded_file)  # Daha önce tanımlanmış fonksiyon
    zaman_serisi = zaman_serisi_olustur(data)
    ekleyiciler = encoder_ve_ekleyici_olustur()
    gecmis_bagimsiz = gecmis_bagimsiz_hazirla(data)

    # Gelecek bağımsız değişkenler için dosya seçimi
    uploaded_future_file = st.file_uploader("Gelecek bağımsız değişken verilerini yükleyin (CSV formatında)", type=["csv"])
    if uploaded_future_file:
        gelecek_bagimsiz = gelecek_bagimsiz_hazirla(gecmis_bagimsiz, uploaded_future_file)
else:
    st.warning("Lütfen bir CSV dosyası yükleyin.")
