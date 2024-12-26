import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from darts.timeseries import TimeSeries
from darts.models import TFTModel
from darts.explainability import TFTExplainer

st.title('🎈 Hava Kirliliği Tahmini Uygulaması')

st.info('Bu uygulama derin öğrenme modeli ile tahmin yapar!')

# Veri Yükleme
st.header("Veri Yükleme")
uploaded_file = st.file_uploader("Hava Kalitesi Verisi Yükleyin (CSV formatında)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col="ds", parse_dates=True)
    st.write("Yüklenen Veri:")
    st.write(data.head())

# Görselleştirme
    st.subheader("Veri Görselleştirme")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['y'])
    ax.set_title("Günlük PM10 Değerleri")
    st.pyplot(fig)
# Model Eğitimi
    st.header("Model Eğitimi")
    if st.button("Modeli Eğit"):
        # Örnek parametrelerle model eğitimi
        model = TFTModel(
            input_chunk_length=30,
            output_chunk_length=7,
            hidden_size=32,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=32,
            lstm_layers=1,
            n_epochs=20,
            use_static_covariates=True,
            pl_trainer_kwargs={'accelerator': 'gpu', "devices": [0]},
        )
        zaman_serisi = TimeSeries.from_dataframe(data, value_cols="y")
        model.fit(zaman_serisi)
        st.success("Model eğitimi tamamlandı!")

        # Tahmin
        st.header("Tahmin Sonuçları")
        tahmin = model.predict(n=7, series=zaman_serisi)
        st.write("Tahmin Edilen Değerler:")
        st.write(tahmin.pd_series())

        # Tahmin Görselleştirme
        fig, ax = plt.subplots()
        ax.plot(zaman_serisi.time_index, zaman_serisi.values(), label="Gerçek")
        ax.plot(tahmin.time_index, tahmin.values(), label="Tahmin", linestyle='dashed')
        ax.set_title("PM10 Tahmini")
        ax.legend()
        st.pyplot(fig)
