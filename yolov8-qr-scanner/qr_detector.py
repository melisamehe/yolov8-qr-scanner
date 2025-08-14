from ultralytics import YOLO
import cv2
from pyzbar.pyzbar import decode

# Modeli yükle
model = YOLO('/home/melisa/Downloads/iyi.pt')  # Tam dosya yolu

# Kamerayı aç
cap = cv2.VideoCapture(0)  # Dahili kamera: 0, harici kamera: 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile QR tespiti
    results = model.predict(source=frame, device='cpu', imgsz=640, conf=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    annotated_frame = results[0].plot()

    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        width = x2 - x1
        height = y2 - y1

        # Biraz genişletilmiş crop bölgesi
        x1 = max(0, x1 - width // 5)
        y1 = max(0, y1 - height // 5)
        x2 = min(frame.shape[1], x2 + width // 5)
        y2 = min(frame.shape[0], y2 + height // 5)

        qr_crop = frame[y1:y2, x1:x2]

        # Yakınlaştır
        zoom_factor = 4
        qr_zoomed = cv2.resize(qr_crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

        # QR kod çözme
        decoded_objects = decode(qr_zoomed)
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            print(f"QR Code Data: {qr_data}")
            cv2.putText(annotated_frame, qr_data, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Sonucu göster
    cv2.imshow('QR Code Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
