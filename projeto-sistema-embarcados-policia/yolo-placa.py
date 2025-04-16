import os
import re

import cv2
import pytesseract
from ultralytics import YOLO


def valida_texto(text):
    if len(text) < 7:  # Verifica se o comprimento total é ao menos 7 (4 letras + 3 números)
        return False
    if len(text) > 10:
        return False
    if len(re.findall(r'[a-zA-Z]', text)) < 4:  # Verifica se há pelo menos 4 letras
        return False
    if len(re.findall(r'\d', text)) < 3:  # Verifica se há pelo menos 3 números
        return False
    return True

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Pasta de saída
save_dir = 'placas_crops'
os.makedirs(save_dir, exist_ok=True)

# Carrega modelo
model = YOLO('placa-veicular-model.pt')

# Use 0 para webcam ou 'video.mp4' para arquivo de vídeo
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rua1.mp4')

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = frame[y1:y2, x1:x2]

        # --- OCR ---
        # Configurações: PSM 7 (uma linha), whitelist só A–Z e 0–9
        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(crop, config=config).strip()

        if not valida_texto(text):
            continue

        print(f"[{counter:04d}] Placa reconhecida:", text)

        # Salva imagem e TXT
        img_path  = os.path.join(save_dir, f'placa_{counter:04d}.jpg')
        txt_path  = os.path.join(save_dir, f'placa_{counter:04d}.txt')
        cv2.imwrite(img_path, crop)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        counter += 1
        # desenha retângulo pra debug
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow('Detecção + OCR', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
