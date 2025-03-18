from ultralytics import YOLO

# Load your vehicle plate detection model
model = YOLO('placa-veicular-model.pt')

windows_ip = "172.20.10.5"  # ← Replace with your actual IP
stream_url = f"http://{windows_ip}:8080/video"

# Run inference on the video stream
results = model(source=stream_url, show=True, conf=0.4, save=True)