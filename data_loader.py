import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(img_size=(224, 224), batch_size=32):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # bir üst klasör
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    valid_generator = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_generator = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, valid_generator, test_generator
