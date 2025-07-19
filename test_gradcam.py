from model_builder import build_model
from gradcam import display_gradcam
from tensorflow.keras.models import load_model

model = build_model(num_classes=6)



model = load_model("my_model.keras")



image_path = "/Users/edanurunal/Desktop/NEU Metal Surface Defects Data/test/Inclusion/In_105.bmp"


last_conv_layer = "Conv_1"


display_gradcam(image_path, model, last_conv_layer_name=last_conv_layer)
