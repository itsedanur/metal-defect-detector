import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# 🔧 Modeli tam yoldan yükle
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "my_model.keras")
model = load_model(model_path)

CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

cap = cv2.VideoCapture(0)  # Kamera aç

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    label = f"{predicted_class} ({confidence * 100:.1f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Canlı Hata Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
print("📷 Kamera kapatıldı. Uygulama sonlandırıldı.")
