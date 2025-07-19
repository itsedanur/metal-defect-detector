#  Metal Defect Detector with Grad-CAM

Bu proje, metal yüzeylerdeki üretim hatalarını tespit etmek için derin öğrenme (CNN) kullanır ve sonuçları Grad-CAM ile görselleştirir.

##  Özellikler
- CNN ile sınıflandırma (6 sınıf)
- Test görselleriyle tahmin
- Grad-CAM ile ısı haritası
- Streamlit arayüzü
- Gerçek zamanlı kamera desteği (OpenCV)

##  Nasıl çalıştırılır

```bash
python train.py
streamlit run streamlit_app.py
python realtime_camera_predict.py
