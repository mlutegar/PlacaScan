from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
import os

# Create output directory
if not os.path.exists('yolo_plate_ocr'):
    os.makedirs('yolo_plate_ocr')

# Load the YOLO model
print("Loading YOLO model...")
try:
    model = YOLO('placa-veicular-model.pt')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Load the image
print("Loading image...")
image = cv2.imread('placas.jpg')
if image is None:
    print("Error: Could not load the image 'placas.jpg'")
    exit()
else:
    print(f"Image loaded successfully. Size: {image.shape}")

# Save the original image
cv2.imwrite('yolo_plate_ocr/original_image.jpg', image)

# Run the YOLO model to detect license plates
print("Running YOLO detection...")
results = model(image)

# Check if any plates were detected
found_plates = False

for i, result in enumerate(results):
    boxes = result.boxes
    
    if len(boxes) == 0:
        print("No license plates detected by YOLO model")
        continue
    
    found_plates = True
    print(f"Found {len(boxes)} license plates")
    
    # Create a copy of the original image for visualization
    vis_image = image.copy()
    
    # Process each detected plate
    for j, box in enumerate(boxes):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])
        
        print(f"Plate #{j+1} - Confidence: {confidence:.4f}, Coordinates: ({x1}, {y1}, {x2}, {y2})")
        
        # Draw rectangle on the visualization image
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, f"Plate #{j+1}: {confidence:.2f}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add padding to the crop
        padding = 5
        y1_pad = max(0, y1 - padding)
        y2_pad = min(image.shape[0], y2 + padding)
        x1_pad = max(0, x1 - padding)
        x2_pad = min(image.shape[1], x2 + padding)
        
        # Crop the license plate with padding
        plate = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        
        # Save the cropped plate
        crop_path = f'yolo_plate_ocr/plate_{j+1}_crop.jpg'
        cv2.imwrite(crop_path, plate)
        print(f"Saved cropped plate to {crop_path}")
        
        # Process cropped plate for OCR
        # First convert to grayscale
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'yolo_plate_ocr/plate_{j+1}_gray.jpg', gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(f'yolo_plate_ocr/plate_{j+1}_thresh.jpg', thresh)
        
        # Apply adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(f'yolo_plate_ocr/plate_{j+1}_adaptive.jpg', adaptive)
        
        # Resize 2x for better OCR
        height, width = gray.shape
        resized2x = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        _, resized2x_thresh = cv2.threshold(resized2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(f'yolo_plate_ocr/plate_{j+1}_resized2x_thresh.jpg', resized2x_thresh)
        
        # OCR configs for Brazilian plates
        print(f"Running OCR on plate #{j+1}...")
        
        ocr_results = []
        ocr_configs = [
            # Try different PSM modes
            ('--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', gray, "Gray - PSM 7"),
            ('--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', gray, "Gray - PSM 8"),
            ('--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', gray, "Gray - PSM 6"),
            
            # Threshold versions
            ('--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', thresh, "Threshold - PSM 7"),
            ('--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', thresh, "Threshold - PSM 8"),
            
            # Adaptive threshold
            ('--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', adaptive, "Adaptive - PSM 7"),
            ('--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', adaptive, "Adaptive - PSM 8"),
            
            # Resized version
            ('--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', resized2x_thresh, "Resized - PSM 7"),
            ('--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', resized2x_thresh, "Resized - PSM 8")
        ]
        
        # Run each OCR configuration
        for config, img, desc in ocr_configs:
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                # Clean the text (remove non-alphanumeric characters)
                clean_text = ''.join(c for c in text if c.isalnum())
                
                if clean_text:
                    print(f"  {desc}: '{clean_text}'")
                    ocr_results.append((desc, clean_text))
            except Exception as e:
                print(f"  Error with {desc}: {e}")
        
        # Save OCR results to a text file
        with open(f'yolo_plate_ocr/plate_{j+1}_ocr_results.txt', 'w') as f:
            f.write(f"OCR Results for Plate #{j+1}\n")
            f.write("=======================\n\n")
            
            if not ocr_results:
                f.write("No text detected with any method!\n")
            else:
                for desc, text in ocr_results:
                    f.write(f"{desc}: '{text}'\n")
        
        # Add best OCR result to the visualization image (if any)
        if ocr_results:
            # Sort by text length (assuming longer results might be more complete)
            ocr_results.sort(key=lambda x: len(x[1]), reverse=True)
            best_method, best_text = ocr_results[0]
            
            cv2.putText(vis_image, f"OCR: {best_text}", (x1, y2 + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the visualization image with all detections and OCR results
    cv2.imwrite('yolo_plate_ocr/plates_detected_with_ocr.jpg', vis_image)

# Check if any plates were found
if not found_plates:
    print("WARNING: No license plates were detected in the image by the YOLO model.")
    print("Possible solutions:")
    print("1. Check if the image contains visible license plates")
    print("2. Verify the YOLO model is correctly trained for this type of license plate")
    print("3. Try adjusting the confidence threshold of the model")

print("\nProcessing complete! Check the 'yolo_plate_ocr' folder for results.")