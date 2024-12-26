import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from darts.timeseries import TimeSeries
from darts.models import TFTModel
from darts.explainability import TFTExplainer
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

