from model_builder import build_model
from gradcam import display_gradcam
from tensorflow.keras.models import load_model
# 1. Modeli Oluştur
model = build_model(num_classes=6)

# 2. Kaydedilmiş Ağırlıkları Yükle


model = load_model("my_model.keras")
 # <<< BURASI YENİ >>>

# 3. Görsel Yolu
image_path = "/Users/edanurunal/Desktop/NEU Metal Surface Defects Data/test/Inclusion/In_105.bmp"

# 4. Son Convolutional Katman Adı (gerekiyorsa katmanları yazdır: for layer in model.layers: print(layer.name))
last_conv_layer = "Conv_1"

# 5. Grad-CAM Göster
display_gradcam(image_path, model, last_conv_layer_name=last_conv_layer)
