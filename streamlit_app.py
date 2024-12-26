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
