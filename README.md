# QR Code Detection with YOLOv8

This project uses a **custom-trained YOLOv8 model** to detect and decode QR codes in real-time from a webcam feed.
<img width="1748" height="978" alt="Screenshot from 2025-08-14 18-44-10" src="https://github.com/user-attachments/assets/7ad5a7cf-1883-4bea-be09-ac3d1cc20474" />


## Features
- Real-time QR code detection using YOLOv8
- Automatic QR code decoding using `pyzbar`
- Annotated live video stream with detected QR code data

---

## Requirements

### Python Dependencies
Install requiredExample Output

When a QR code is detected, the terminal will print: Python packages:
```bash
pip install ultralytics opencv-python pyzbar
```

###System Dependencies

For QR decoding to work on Linux, install:
```bash
sudo apt install libzbar0
```

Model

Place your trained YOLOv8 model (e.g., iyi.pt) in your desired location and update the file path in the code:
```bash
model = YOLO('/home/iyi.pt')
```

Usage

Connect your webcam.

Run the script:
```bash
python qr_detector.py
```



