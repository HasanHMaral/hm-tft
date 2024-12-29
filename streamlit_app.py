import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.explainability import TFTExplainer
import torch

# Başlık ve açıklamalar
st.title('🎈 Hava Kirliliği Tahmini ve Model Analizi Uygulaması')
st.info('Bu uygulama ile modeli eğitebilir, tahmin yapabilir ve modelin davranışını analiz edebilirsiniz!')

# Yıl kodlama fonksiyonu
def yil_kodla(idx):
    return (idx.year - 2000) / 50

# Veri yükleme ve işleme
def yukle_ve_isle(dosya_adi):
    try:
        data = pd.read_csv(dosya_adi, index_col='ds', parse_dates=True)
        data = data.asfreq('d')
        # Veri görselleştirme
        st.subheader("Günlük PM10 Değeri Grafiği")
        fig, ax = plt.subplots(figsize=(10, 6))
        data['y'].plot(title='Günlük PM10 Değeri', ax=ax)
        st.pyplot(fig)
        # Otokorelasyon grafikleri
        st.subheader("Otokorelasyon Grafiği")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(data['y'], lags=100, ax=ax)
        st.pyplot(fig)
        st.subheader("Parsiyel Otokorelasyon Grafiği")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_pacf(data['y'], lags=100, ax=ax)
        st.pyplot(fig)
        return data
    except Exception as e:
        st.error(f"Veri yüklenirken bir hata oluştu: {str(e)}")
        return None

# Zaman serisi oluşturma
def zaman_serisi_olustur(data):
    try:
        zaman_serisi = TimeSeries.from_dataframe(data, value_cols="y")
        st.write("Zaman Serisi:")
        st.write(zaman_serisi)
        return zaman_serisi
    except Exception as e:
        st.error(f"Zaman serisi oluşturulurken bir hata oluştu: {str(e)}")
        return None

# Zaman serisi ölçekleme
def zaman_serisi_olcekle(zaman_serisi):
    try:
        olcekleyici = Scaler()
        trans_zaman_serisi = olcekleyici.fit_transform(zaman_serisi)
        st.write("Zaman Serisi Ölçeklendirilmiş:")
        st.write(trans_zaman_serisi)
        return olcekleyici, trans_zaman_serisi
    except Exception as e:
        st.error(f"Zaman serisi ölçeklenirken bir hata oluştu: {str(e)}")
        return None, None

# Geçmiş bağımsız değişkenler
def gecmis_bagimsiz_hazirla(data):
    try:
        X_gecmis = data.iloc[:, 2:]
        gecmis_bagimsiz = TimeSeries.from_dataframe(X_gecmis)
        st.write("Geçmiş Bağımsız Değişkenler:")
        st.write(gecmis_bagimsiz)
        return gecmis_bagimsiz
    except Exception as e:
        st.error(f"Geçmiş bağımsız değişkenler hazırlanırken bir hata oluştu: {str(e)}")
        return None

# Gelecek bağımsız değişkenler
def gelecek_bagimsiz_hazirla(gecmis_bagimsiz, gelecek_veri_dosyasi):
    try:
        gelecek_veri = pd.read_csv(gelecek_veri_dosyasi, index_col='ds', parse_dates=True)
        X_gelecek = gelecek_veri.iloc[:, 1:]
        bagimsiz_tum = pd.concat([gecmis_bagimsiz.pd_dataframe(), X_gelecek])
        gelecek_bagimsiz = TimeSeries.from_dataframe(bagimsiz_tum)
        st.write("Gelecek Bağımsız Değişkenler:")
        st.write(gelecek_bagimsiz)
        return gelecek_bagimsiz
    except Exception as e:
        st.error(f"Gelecek bağımsız değişkenler hazırlanırken bir hata oluştu: {str(e)}")
        return None

# Bağımsız değişkenleri ölçekleme
def bagimsiz_degiskenleri_olcekle(gecmis_bagimsiz, gelecek_bagimsiz):
    try:
        olcekleyici = Scaler()
        transformed_gecmis_bagimsiz = olcekleyici.fit_transform(gecmis_bagimsiz)
        transformed_gelecek_bagimsiz = olcekleyici.fit_transform(gelecek_bagimsiz)
        st.write("Geçmiş Bağımsız Değişkenler Ölçeklendirilmiş:")
        st.write(transformed_gecmis_bagimsiz)
        st.write("Gelecek Bağımsız Değişkenler Ölçeklendirilmiş:")
        st.write(transformed_gelecek_bagimsiz)
        return olcekleyici, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz
    except Exception as e:
        st.error(f"Bağımsız değişkenler ölçeklenirken bir hata oluştu: {str(e)}")
        return None, None, None

# Model eğitme ve kaydetme
def modeli_egit_ve_kaydet(trans_zaman_serisi, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz):
    try:
        en_iyi_parametre_dict = {
            'output_chunk_length': 30,
            'num_attention_heads': 4,
            'n_epochs': 20,
            'lstm_layers': 1,
            'input_chunk_length': 90,
            'hidden_size': 64,
            'dropout': 0.1,
            'batch_size': 32,
            'use_static_covariates': True,
            'add_encoders': None,
            'pl_trainer_kwargs': {'accelerator': 'cpu', 'devices': 1}
        }
        model = TFTModel(**en_iyi_parametre_dict)
        model.fit(trans_zaman_serisi,
                  past_covariates=transformed_gecmis_bagimsiz,
                  future_covariates=transformed_gelecek_bagimsiz)
        model.save("ayarli_tft_model_cpu.pth")
        st.success("Model başarıyla eğitildi ve kaydedildi!")
    except Exception as e:
        st.error(f"Model eğitilirken bir hata oluştu: {str(e)}")

# Tahmin yapma
def tahmin_yap(model, trans_zaman_serisi, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz, olcekleyici1):
    try:
        tahmin = model.predict(
            n=30,  # Tahmin ufku
            series=trans_zaman_serisi,
            past_covariates=transformed_gecmis_bagimsiz,
            future_covariates=transformed_gelecek_bagimsiz
        )
        tahmin = TimeSeries.pd_series(olcekleyici1.inverse_transform(tahmin)).rename("TFT")
        return tahmin
    except Exception as e:
        st.error(f"Tahmin yapılırken bir hata oluştu: {str(e)}")
        return None

# Model analizi
def modeli_anlamlandir(model):
    try:
        explainer = TFTExplainer(model)
        explainability_results = explainer.explain()

        # Değişken önemini görselleştirme
        st.subheader("Değişken Önemi")
        fig, ax = plt.subplots(figsize=(10, 10))
        explainer.plot_variable_selection(explainability_results, ax=ax)
        st.pyplot(fig)

        # Dikkat mekanizmasını görselleştirme
        st.subheader("Dikkat Mekanizması")
        fig, ax = plt.subplots(figsize=(10, 10))
        explainer.plot_attention(explainability_results, ax=ax, plot_type="time")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Model analizi yapılırken bir hata oluştu: {str(e)}")

# Kullanıcı girişlerini alma ve işlemleri başlatma
uploaded_file = st.file_uploader("Hava kalitesi verisini yükleyin (CSV)", type=["csv"])
if uploaded_file:
    data = yukle_ve_isle(uploaded_file)
    if data is not None:
        zaman_serisi = zaman_serisi_olustur(data)
        if zaman_serisi is not None:
            olcekleyici1, trans_zaman_serisi = zaman_serisi_olcekle(zaman_serisi)

            # Geçmiş bağımsız değişkenler
            gecmis_bagimsiz = gecmis_bagimsiz_hazirla(data)

            # Gelecek bağımsız değişkenler için yükleme
            future_file = st.file_uploader("Gelecek bağımsız değişken verilerini yükleyin (CSV)", type=["csv"])
            if future_file:
                gelecek_bagimsiz = gelecek_bagimsiz_hazirla(gecmis_bagimsiz, future_file)

                olcekleyici2, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz = bagimsiz_degiskenleri_olcekle(
                    gecmis_bagimsiz, gelecek_bagimsiz)

                # Eğitim ve tahmin işlemleri
                if st.button("Modeli Eğit ve Kaydet"):
                    modeli_egit_ve_kaydet(trans_zaman_serisi, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz)
                if st.button("Modeli Yükle ve Tahmin Yap"):
                    model = TFTModel.load("ayarli_tft_model_cpu.pth", map_location="cpu")
                    tahmin = tahmin_yap(model, trans_zaman_serisi, transformed_gecmis_bagimsiz, transformed_gelecek_bagimsiz, olcekleyici1)
                    if tahmin is not None:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(data['y'], label="Gerçek Değerler")
                        ax.plot(tahmin, label="Tahminler", linestyle='dashed')
                        ax.set_xlabel("Zaman")
                        ax.set_ylabel("PM10")
                        ax.set_title("TFT Tahmin")
                        ax.legend()
                        st.pyplot(fig)
                    modeli_anlamlandir(model)
