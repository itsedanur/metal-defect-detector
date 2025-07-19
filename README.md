# 🔍 Metal Defect Detector with Grad-CAM

Bu proje, metal yüzeylerdeki üretim hatalarını tespit eden bir derin öğrenme (CNN) sistemidir. Grad-CAM sayesinde görselleştirme yapılabilir.  

## Özellikler
-  6 sınıflı metal yüzey kusur sınıflandırması
-  Grad-CAM ile ısı haritası görselleştirme
-  Streamlit ile görsel arayüz
-  Gerçek zamanlı kamera ile canlı tahmin (OpenCV)

##  Dosyalar
- `train.py` – Modeli eğitir ve kaydeder (`my_model.keras`)
- `streamlit_app.py` – Web arayüzü ile tahmin ve Grad-CAM
- `realtime_camera_predict.py` – Webcam ile tahmin
- `gradcam.py` – Grad-CAM üretici
- `data_loader.py` – Eğitim/validasyon/test veri yükleyici
- `model_builder.py` – CNN mimarisi oluşturucu
- `generate_gradcam_all.py` – Tüm test görselleri için Grad-CAM oluşturur

## Gerekli Kurulumlar
```bash
pip install tensorflow opencv-python streamlit matplotlib pillow

Kullanılan Teknolojiler

TensorFlow / Keras
Streamlit
OpenCV
Grad-CAM
