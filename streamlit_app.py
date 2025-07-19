import streamlit as st
import numpy as np
import os
from keras.models import load_model
from PIL import Image
from gradcam import display_gradcam

# 📌 Modeli sadece bir kez yükle
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "my_model.keras")
    return load_model(model_path)

# ✅ Sınıf isimleri
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# 🌟 Başlık
st.title("🔍 Metal Yüzey Hata Tespiti ve Grad-CAM Görselleştirme")

# 📦 Modeli yükle
model = load_trained_model()

# 📤 Kullanıcıdan görsel yüklemesini iste
uploaded_file = st.file_uploader("Bir test görüntüsü yükleyin", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # 📷 Görseli oku
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # 🔍 Modelin beklediği şekle dönüştür
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 📈 Tahmin yap
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.success(f"Tahmin: **{predicted_class}** (%{confidence * 100:.2f} güven)")

    # 🌡️ Grad-CAM çalıştır ve sonucu göster
    temp_path = "temp_uploaded_img.jpg"
    image.save(temp_path)

    display_gradcam(temp_path, model, last_conv_layer_name="Conv_1", save_path="gradcam_output.png")

    st.image("gradcam_output.png", caption="Grad-CAM Isı Haritası", use_column_width=True)
