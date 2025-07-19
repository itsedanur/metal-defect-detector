import os
from gradcam import display_gradcam
from tensorflow.keras.models import load_model

model = load_model("my_model.keras")
test_image_dir = "/Users/edanurunal/Desktop/NEU Metal Surface Defects Data/test"
save_dir = "gradcam_results"
os.makedirs(save_dir, exist_ok=True)

for class_name in os.listdir(test_image_dir):
    class_dir = os.path.join(test_image_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        try:
            output_path = os.path.join(save_dir, f"{class_name}_{img_file.replace('.bmp', '')}_gradcam.png")
            display_gradcam(img_path, model, last_conv_layer_name="Conv_1", save_path=output_path)
            print(f"Kaydedildi: {output_path}")
        except Exception as e:
            print(f"HATA: {img_path} - {e}")
