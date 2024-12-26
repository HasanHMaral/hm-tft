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

X_gecmis = data.iloc[:, 2:]  # İlk iki sütun hariç diğer sütunları al
st.write("Geçmiş Bağımsız Değişkenler (İlk 5 Satır):")
st.write(X_gecmis.head())
    try:# TimeSeries nesnesi oluşturma
    gecmis_bagimsiz = TimeSeries.from_dataframe(X_gecmis)
    st.write("Geçmiş Bağımsız Değişkenlerin TimeSeries Nesnesi:")
    st.write(gecmis_bagimsiz)
    except Exception as e:
    st.error(f"TimeSeries nesnesi oluşturulurken bir hata oluştu: {str(e)}")


# Gelecek Bağımsız Değişkenlerin Hazırlanması
st.subheader("Gelecek Bağımsız Değişkenlerin Hazırlanması")

# Gelecek verisini yüklemek için dosya yükleme bileşeni
uploaded_future = st.file_uploader("Gelecek Verisini Yükleyin (CSV formatında)", type="csv")

if uploaded_future:
    # Gelecek veriyi yükleme
    gelecek_veri = pd.read_csv(uploaded_future, index_col="ds", parse_dates=True)
    
    # Gelecek bağımsız değişkenleri seçme
    X_gelecek = gelecek_veri.iloc[:, 1:]  # İlk sütun hariç diğer sütunları al
    st.write("Gelecek Bağımsız Değişkenler:")
    st.write(X_gelecek.head())

    # TimeSeries nesnesi oluşturma
    try:
        gelecek_bagimsiz = TimeSeries.from_dataframe(X_gelecek)
        st.write("Gelecek Bağımsız Değişkenlerin TimeSeries Nesnesi:")
        st.write(gelecek_bagimsiz)

         # Ölçekleyiciler
        olcekleyici1 = Scaler()
        olcekleyici2 = Scaler()

        # Zaman serisini ölçekleme
        trans_zaman_serisi = olcekleyici1.fit_transform(zaman_serisi)
        st.write("Zaman Serisi Ölçeklendirilmiş:")
        st.write(trans_zaman_serisi)

        # Geçmiş bağımsız değişkenleri ölçekleme
        transformed_gecmis_bagimsiz = olcekleyici2.fit_transform(gecmis_bagimsiz)
        st.write("Geçmiş Bağımsız Değişkenler Ölçeklendirilmiş:")
        st.write(transformed_gecmis_bagimsiz)

        # Gelecek bağımsız değişkenleri ölçekleme
        transformed_gelecek_bagimsiz = olcekleyici2.fit_transform(gelecek_bagimsiz)
        st.write("Gelecek Bağımsız Değişkenler Ölçeklendirilmiş:")
        st.write(transformed_gelecek_bagimsiz)   

    except Exception as e:
        st.error(f"TimeSeries nesnesi oluşturulurken bir hata oluştu: {str(e)}")
# Eksik verileri kontrol etme
    if transformed_gecmis_bagimsiz.has_missing_values():
        st.warning("Geçmiş bağımsız değişkenlerde eksik değerler bulundu. Eksik veriler dolduruluyor...")
        transformed_gecmis_bagimsiz = transformed_gecmis_bagimsiz.fill_missing_values()
        st.success("Geçmiş bağımsız değişkenlerdeki eksik değerler dolduruldu.")

    if transformed_gelecek_bagimsiz.has_missing_values():
        st.warning("Gelecek bağımsız değişkenlerde eksik değerler bulundu. Eksik veriler dolduruluyor...")
        transformed_gelecek_bagimsiz = transformed_gelecek_bagimsiz.fill_missing_values()
        st.success("Gelecek bağımsız değişkenlerdeki eksik değerler dolduruldu.")

    # Kontrol sonrası bilgi mesajı
    st.write("Eksik veri kontrolü tamamlandı.")

# Model Oluşturma ve Çapraz Doğrulama
st.subheader("TFT Modeli Oluşturma ve Eğitim")

# TFT model parametreleri için kullanıcıdan giriş alımı
input_chunk_length = st.number_input("Input Chunk Length:", min_value=10, max_value=365, value=90, step=10)
output_chunk_length = st.number_input("Output Chunk Length:", min_value=1, max_value=60, value=30, step=1)
hidden_size = st.number_input("Hidden Size:", min_value=4, max_value=128, value=16, step=4)
lstm_layers = st.number_input("LSTM Layers:", min_value=1, max_value=5, value=2, step=1)
num_attention_heads = st.number_input("Number of Attention Heads:", min_value=1, max_value=8, value=4, step=1)
dropout = st.slider("Dropout Rate:", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
batch_size = st.number_input("Batch Size:", min_value=8, max_value=128, value=64, step=8)
n_epochs = st.number_input("Number of Epochs:", min_value=1, max_value=100, value=10, step=1)
use_static_covariates = st.checkbox("Use Static Covariates", value=True)
accelerator = st.selectbox("Trainer Accelerator:", ["gpu", "cpu"], index=0)
devices = st.number_input("Number of Devices:", min_value=1, max_value=4, value=1, step=1)

if st.button("Modeli Eğit"):
    try:
        # TFT modeli oluşturma
        model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=n_epochs,
            add_encoders=ekleyiciler,  # Önceden tanımlanmış ekleyiciler
            use_static_covariates=use_static_covariates,
            pl_trainer_kwargs={'accelerator': accelerator, 'devices': [devices]}
        )

        # Modeli eğitim verileriyle eğitme
        model.fit(
            trans_zaman_serisi,
            past_covariates=transformed_gecmis_bagimsiz,
            future_covariates=transformed_gelecek_bagimsiz
        )
        st.success("Model başarıyla eğitildi!")

    except Exception as e:
        st.error(f"Model oluşturulurken veya eğitilirken bir hata oluştu: {str(e)}")
