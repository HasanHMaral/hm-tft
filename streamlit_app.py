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
def yil_kodla(idx):
    return (idx.year - 2000) / 50
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
def zaman_serisi_olcekle(zaman_serisi):
    """
    Zaman serisini ölçekler.

    Args:
        zaman_serisi (TimeSeries): Zaman serisi nesnesi.

    Returns:
        Scaler, TimeSeries: Ölçekleyici nesnesi ve ölçeklenmiş zaman serisi.
    """
    olcekleyici = Scaler()
    trans_zaman_serisi = olcekleyici.fit_transform(zaman_serisi)
    st.write("Zaman Serisi Ölçeklendirilmiş:")
    st.write(trans_zaman_serisi)
    return olcekleyici, trans_zaman_serisi

def bagimsiz_degiskenleri_olcekle(gecmis_bagimsiz, gelecek_bagimsiz):
    """
    Geçmiş ve gelecek bağımsız değişkenleri ölçekler.

    Args:
        gecmis_bagimsiz (TimeSeries): Geçmiş bağımsız değişkenler.
        gelecek_bagimsiz (TimeSeries): Gelecek bağımsız değişkenler.

    Returns:
        Scaler, TimeSeries, TimeSeries: Ölçekleyici nesnesi ve ölçeklenmiş bağımsız değişkenler.
    """
    olcekleyici = Scaler()
    transformed_gecmis_bagimsiz = olcekleyici.fit_transform(gecmis_bagimsiz)
    st.write("Geçmiş Bağımsız Değişkenler Ölçeklendirilmiş:")
    st.write(transformed_gecmis_bagimsiz)

    transformed_gelecek_bagimsiz = olcekleyici.fit_transform(gelecek_bagimsiz)
    st.write("Gelecek Bağımsız Değişkenler Ölçeklendirilmiş:")
    st.write(transformed_gelecek_bagimsiz)

    return olcekleyici, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz
def encoder_ve_ekleyici_olustur():
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
    return ekleyiciler

# Yükleme öncesi ekleyiciyi oluştur
ekleyiciler = encoder_ve_ekleyici_olustur()
# Eğitilmiş modelin yolu
model_path = "tuned_tft_model.pth"  # GitHub deposundaki model dosyası yolu

# Eğitilmiş modelin yüklenmesi
def egitilmis_modeli_yukle(model_path):
    try:
        # Modeli CPU ortamında yüklemek için map_location ayarı
        model = TFTModel.load(model_path, map_location="cpu")
        st.success("Eğitilmiş model başarıyla yüklendi!")
        return model
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {str(e)}")
        return None

uploaded_file = st.file_uploader("Lütfen hava kalitesi verisini yükleyin (CSV formatında)", type=["csv"])
if uploaded_file is not None:
    # Veri yükleme ve işleme
    data = yukle_ve_isle(uploaded_file)  # Daha önce tanımlanan veri yükleme ve işleme fonksiyonu

    # Zaman serisi oluşturma
    zaman_serisi = zaman_serisi_olustur(data)

    # Encoder ve ekleyiciler oluşturma
    ekleyiciler = encoder_ve_ekleyici_olustur()

    # Geçmiş bağımsız değişkenleri hazırlama
    gecmis_bagimsiz = gecmis_bagimsiz_hazirla(data)

    # Gelecek bağımsız değişkenler için dosya yükleme
    uploaded_future_file = st.file_uploader("Gelecek bağımsız değişken verilerini yükleyin (CSV formatında)", type=["csv"])
    if uploaded_future_file:
        # Gelecek bağımsız değişkenleri hazırlama
        gelecek_bagimsiz = gelecek_bagimsiz_hazirla(gecmis_bagimsiz, uploaded_future_file)

        # Zaman serisini ve bağımsız değişkenleri ölçekleme
        olcekleyici1, trans_zaman_serisi = zaman_serisi_olcekle(zaman_serisi)
        olcekleyici2, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz = bagimsiz_degiskenleri_olcekle(
            gecmis_bagimsiz, gelecek_bagimsiz
        )

        st.success("Tüm veri işleme ve ölçeklendirme adımları başarıyla tamamlandı!")

    if st.button("Modeli Yükle"):
        model = egitilmis_modeli_yukle(model_path)

    if model:
        # Tahmin yapma işlemi
        forecasting_horizon = 30
        predictions = model.predict(
            n=forecasting_horizon,
            series=trans_zaman_serisi,
            past_covariates=transformed_gecmis_bagimsiz,
            future_covariates=transformed_gelecek_bagimsiz
        )

        # Tahminleri görselleştirme
        st.subheader("Tahmin Sonuçları")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(trans_zaman_serisi.unscale().pd_series(), label="Gerçek Değerler")
        ax.plot(predictions.unscale().pd_series(), label="Tahminler", linestyle="--")
        ax.set_title("Tahmin Sonuçları")
        ax.legend()
        st.pyplot(fig)
       
else:
    st.warning("Lütfen bir CSV dosyası yükleyin.")
