from model_builder import build_model
from data_loader import get_data_generators

train_gen, valid_gen, test_gen = get_data_generators()

print(f"Eğitimde {train_gen.samples} görüntü var.")
print(f"Doğrulamada {valid_gen.samples} görüntü var.")
print(f"Testte {test_gen.samples} görüntü var.")


model = build_model(num_classes=6)


history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=5,
    verbose=1
)

loss, acc = model.evaluate(test_gen)
print(f"\nTest doğruluğu: {acc:.4f} | Test kaybı: {loss:.4f}")


model.save("my_model.keras")


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc, label='Eğitim Doğruluğu')
    axes[0].plot(epochs_range, val_acc, label='Doğrulama Doğruluğu')
    axes[0].set_title('Doğruluk (Accuracy)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Oran')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].legend()

    axes[1].plot(epochs_range, loss, label='Eğitim Kaybı')
    axes[1].plot(epochs_range, val_loss, label='Doğrulama Kaybı')
    axes[1].set_title('Kayıp (Loss)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Değer')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("egitim_grafikleri.png")  
    print("Grafik kaydedildi: egitim_grafikleri.png")  
    plt.show()

plot_training_history(history)


from gradcam import display_gradcam

image_path = "/Users/edanurunal/Desktop/NEU Metal Surface Defects Data/test/Inclusion/In_105.bmp"
last_conv_layer = "Conv_1"  
display_gradcam(image_path, model, last_conv_layer_name=last_conv_layer)
