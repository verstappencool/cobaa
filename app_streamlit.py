import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Nama-nama kelas (kelas-kelas yang sudah dilatih pada model)
class_names = ['Cheetah', 'Jaguar', 'Jaguar_Hitam', 'Leopard', 'Lion', 'Puma', 'Tiger']

# Memuat model yang telah disimpan
model = load_model('my_model-2.h5')

# Fungsi prediksi
@tf.function
def predict_image(img_array):
    return model(img_array)

# Data taksonomi
taxonomy_data = {
    "Cheetah": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Acinonyx, Species: Acinonyx jubatus",
        "deskripsi": "Cheetah adalah kucing besar yang terkenal karena kecepatannya."
    },
    "Jaguar": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera onca",
        "deskripsi": "Jaguar adalah kucing besar yang hidup di hutan hujan tropis Amerika."
    },
    "Jaguar_Hitam": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera onca (Varian melanistik)",
        "deskripsi": "Jaguar Hitam adalah varian melanistik dari jaguar yang memiliki warna hitam pekat."
    },
    "Leopard": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera pardus",
        "deskripsi": "Leopard adalah kucing besar yang hidup di Afrika dan Asia, dikenal karena corak belangnya."
    },
    "Lion": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera leo",
        "deskripsi": "Lion adalah kucing besar yang dikenal dengan surai di sekitar lehernya, simbol kekuatan."
    },
    "Puma": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Puma, Species: Puma concolor",
        "deskripsi": "Puma, atau cougar, adalah kucing besar yang tersebar di seluruh Amerika."
    },
    "Tiger": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera tigris",
        "deskripsi": "Tiger adalah kucing besar yang dikenal dengan garis-garis tubuhnya, hidup di Asia."
    }
}

# Aplikasi Streamlit
st.title("Klasifikasi Kucing - Keluarga Felidae")
st.markdown(
    """
    Aplikasi ini digunakan untuk mengklasifikasikan gambar kucing besar dari keluarga taksonomi **Felidae**.
    Upload gambar, dan aplikasi akan menampilkan prediksi spesies, taksonomi, serta deskripsi singkat.
    """
)

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Proses gambar
    img = Image.open(uploaded_file)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = predict_image(img_array)
    predictions_np = predictions.numpy()[0]
    predicted_class_index = np.argmax(predictions_np)
    predicted_class = class_names[predicted_class_index]
    predicted_prob = np.max(predictions_np)

    # Tampilkan hasil
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)
    st.subheader("Hasil Prediksi")
    st.write(f"**Spesies yang terdeteksi:** {predicted_class}")
    st.write(f"**Probabilitas:** {predicted_prob*100:.2f}%")

    # Tampilkan taksonomi
    species_info = taxonomy_data.get(predicted_class, {})
    if species_info:
        st.markdown("### Informasi Taksonomi dan Deskripsi")
        st.write(f"**Taksonomi:** {species_info['taksonomi']}")
        st.write(f"**Deskripsi:** {species_info['deskripsi']}")

    # Probabilitas untuk setiap kelas
    st.markdown("### Probabilitas Semua Kelas")
    for i, class_name in enumerate(class_names):
        if i < len(predictions_np):
            st.write(f"{class_name}: {predictions_np[i] * 100:.2f}%")

st.markdown("---")
st.info("Pastikan gambar yang diunggah merupakan anggota keluarga Felidae untuk hasil yang lebih akurat.")
