from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Try importing EasyOCR
try:
    import easyocr
    print("EasyOCR successfully imported")
except ImportError:
    print("EasyOCR not found. Installing...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
    import easyocr
    print("EasyOCR installed successfully")

# Create output directory
if not os.path.exists('yolo_easyocr_threshold'):
    os.makedirs('yolo_easyocr_threshold')

# Step 1: Initialize EasyOCR reader (use Portuguese for Brazilian plates)
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en', 'pt'])  # Use both English and Portuguese for better results
print("✓ EasyOCR reader initialized")

# Step 2: Load the YOLO model
print("Loading YOLO model...")
try:
    model = YOLO('placa-veicular-model.pt')  # Load your custom license plate model
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    exit(1)

# Step 3: Load the image
print("Loading image...")
image_path = 'placas.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"✗ Error: Could not load the image: {image_path}")
    exit(1)
else:
    print(f"✓ Image loaded successfully. Dimensions: {image.shape}")
    # Save a copy of the original
    cv2.imwrite('yolo_easyocr_threshold/original_image.jpg', image)

# Step 4: Run the YOLO model for detection
print("\nRunning YOLO detection...")
results = model(image)

# Create a visualization image
detection_image = image.copy()

# Check if detection produced any results
if not results or len(results) == 0:
    print("✗ No detection results returned by the model")
    exit(1)

# Function to run EasyOCR with threshold filtering
def run_easyocr_with_threshold(img, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Run EasyOCR with multiple confidence thresholds
    
    Args:
        img: Input image
        thresholds: List of confidence threshold values to try
        
    Returns:
        Dictionary of results for each threshold
    """
    threshold_results = {}
    
    # Get raw EasyOCR results
    raw_results = reader.readtext(img)
    
    # Process for each threshold
    for threshold in thresholds:
        # Filter by confidence score
        filtered_results = [res for res in raw_results if res[2] >= threshold]
        
        # Compile results
        if filtered_results:
            # Extract text and scores
            texts = []
            for bbox, text, score in filtered_results:
                # Clean the text (keep only alphanumeric)
                clean_text = ''.join(c for c in text if c.isalnum())
                if clean_text:  # Only add non-empty text
                    texts.append((clean_text, score))
            
            # Combine all detected text
            if texts:
                # Sort by confidence (highest first)
                texts.sort(key=lambda x: x[1], reverse=True)
                # Extract combined text and individual confidences
                combined_text = ''.join([t[0] for t in texts])
                text_details = texts
                
                threshold_results[threshold] = {
                    'combined_text': combined_text,
                    'text_details': text_details,
                    'raw_detections': filtered_results
                }
    
    return threshold_results, raw_results

# Process all detected plates
plates_detected = 0
for result_idx, result in enumerate(results):
    boxes = result.boxes
    
    if len(boxes) == 0:
        print(f"✗ No license plates detected in result {result_idx+1}")
        continue
    
    print(f"✓ Found {len(boxes)} license plate(s) in result {result_idx+1}")
    plates_detected += len(boxes)
    
    # Step 5: Process each detected license plate
    for plate_idx, box in enumerate(boxes):
        # Extract coordinates
        try:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            print(f"\nProcessing plate #{plate_idx+1} (YOLO Confidence: {confidence:.2f})")
            print(f"Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Draw the detection on the visualization image
            cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Plate {plate_idx+1}: {confidence:.2f}"
            cv2.putText(detection_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Step 6: Add padding and crop the license plate from the original image
            padding = 5  # pixels of padding around the detected plate
            y1_pad = max(0, y1 - padding)
            y2_pad = min(image.shape[0], y2 + padding)
            x1_pad = max(0, x1 - padding)
            x2_pad = min(image.shape[1], x2 + padding)
            
            # Perform the actual cropping
            cropped_plate = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            
            # Check if crop was successful
            if cropped_plate.size == 0 or cropped_plate.shape[0] == 0 or cropped_plate.shape[1] == 0:
                print(f"✗ Error: Cropped plate is empty. Check coordinates.")
                continue
                
            print(f"✓ Cropped plate dimensions: {cropped_plate.shape}")
            
            # Save the cropped plate
            crop_path = f'yolo_easyocr_threshold/plate_{plate_idx+1}_cropped.jpg'
            cv2.imwrite(crop_path, cropped_plate)
            print(f"✓ Saved cropped plate to: {crop_path}")
            
            # Step 7: Preprocess the cropped plate for OCR
            # Create a figure to show all preprocessing methods
            plt.figure(figsize=(15, 10))
            processed_images = []
            
            # Original cropped plate
            plt.subplot(3, 3, 1)
            plt.imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
            plt.title("Original Crop")
            plt.axis('off')
            processed_images.append(("Original", cropped_plate))
            
            # Grayscale conversion
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            plt.subplot(3, 3, 2)
            plt.imshow(gray_plate, cmap='gray')
            plt.title("Grayscale")
            plt.axis('off')
            processed_images.append(("Grayscale", gray_plate))
            cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_gray.jpg', gray_plate)
            
            # Otsu's Thresholding
            _, otsu_thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plt.subplot(3, 3, 3)
            plt.imshow(otsu_thresh, cmap='gray')
            plt.title("Otsu Threshold")
            plt.axis('off')
            processed_images.append(("Otsu", otsu_thresh))
            cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_otsu.jpg', otsu_thresh)
            
            # Adaptive Thresholding
            adaptive_thresh = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
            plt.subplot(3, 3, 4)
            plt.imshow(adaptive_thresh, cmap='gray')
            plt.title("Adaptive Threshold")
            plt.axis('off')
            processed_images.append(("Adaptive", adaptive_thresh))
            cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_adaptive.jpg', adaptive_thresh)
            
            # Bilateral Filter (reduces noise while preserving edges)
            bilateral = cv2.bilateralFilter(gray_plate, 11, 17, 17)
            plt.subplot(3, 3, 5)
            plt.imshow(bilateral, cmap='gray')
            plt.title("Bilateral Filter")
            plt.axis('off')
            processed_images.append(("Bilateral", bilateral))
            cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_bilateral.jpg', bilateral)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray_plate, -1, kernel)
            plt.subplot(3, 3, 6)
            plt.imshow(sharpened, cmap='gray')
            plt.title("Sharpened")
            plt.axis('off')
            processed_images.append(("Sharpened", sharpened))
            cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_sharpened.jpg', sharpened)
            
            # Resized (2x)
            height, width = gray_plate.shape
            resized2x = cv2.resize(gray_plate, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
            plt.subplot(3, 3, 7)
            plt.imshow(resized2x, cmap='gray')
            plt.title("Resized 2x")
            plt.axis('off')
            processed_images.append(("Resized2x", resized2x))
            cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_resized2x.jpg', resized2x)
            
            # Inverted
            inverted = cv2.bitwise_not(gray_plate)
            plt.subplot(3, 3, 8)
            plt.imshow(inverted, cmap='gray')
            plt.title("Inverted")
            plt.axis('off')
            processed_images.append(("Inverted", inverted))
            cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_inverted.jpg', inverted)
            
            # Save the preprocessing visualization
            plt.tight_layout()
            plt.savefig(f'yolo_easyocr_threshold/plate_{plate_idx+1}_preprocessing.jpg')
            plt.close()
            
            # Step 8: Run EasyOCR with threshold filtering on each processed version
            print("\nRunning EasyOCR with confidence thresholds...")
            
            # Define thresholds to try
            confidence_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]
            
            # Collect all results to find the best
            all_threshold_results = []
            
            for desc, img in processed_images:
                print(f"Processing {desc}...")
                try:
                    # Apply EasyOCR with various thresholds
                    threshold_results, raw_results = run_easyocr_with_threshold(img, confidence_thresholds)
                    
                    # Save raw detection visualization
                    if raw_results:
                        result_img = img.copy()
                        if len(result_img.shape) == 2:  # Convert grayscale to color for visualization
                            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
                            
                        for detection in raw_results:
                            bbox, text, score = detection
                            # Convert points to integers
                            points = np.array(bbox).astype(np.int32)
                            # Draw bounding box
                            cv2.polylines(result_img, [points], True, (0, 255, 0), 2)
                            # Add text and confidence
                            label = f"{text} ({score:.2f})"
                            cv2.putText(result_img, label, (points[0][0], points[0][1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save visualization with all detections
                        cv2.imwrite(f'yolo_easyocr_threshold/plate_{plate_idx+1}_{desc.lower()}_all_detections.jpg', result_img)
                    
                    # Log and collect results for each threshold
                    for threshold, results in threshold_results.items():
                        combined_text = results['combined_text']
                        print(f"  Threshold {threshold}: '{combined_text}'")
                        
                        # Store result
                        all_threshold_results.append((desc, threshold, combined_text, results['text_details']))
                    
                except Exception as e:
                    print(f"  Error with {desc}: {e}")
            
            # Save OCR results to a text file
            with open(f'yolo_easyocr_threshold/plate_{plate_idx+1}_threshold_results.txt', 'w') as f:
                f.write(f"EasyOCR Threshold Results for Plate #{plate_idx+1}\n")
                f.write("============================================\n\n")
                
                if not all_threshold_results:
                    f.write("No text detected with any preprocessing method or threshold!\n")
                else:
                    # Group by preprocessing method
                    methods = {}
                    for desc, threshold, text, details in all_threshold_results:
                        if desc not in methods:
                            methods[desc] = []
                        methods[desc].append((threshold, text, details))
                    
                    # Write results by method
                    for desc, results in methods.items():
                        f.write(f"\n{desc}:\n")
                        f.write("-" * len(desc) + "\n")
                        
                        for threshold, text, details in results:
                            f.write(f"  Threshold {threshold}: '{text}'\n")
                            # Write individual character confidences
                            if details:
                                f.write("    Character confidences:\n")
                                for char_text, confidence in details:
                                    f.write(f"      '{char_text}': {confidence:.4f}\n")
                        f.write("\n")
            
            # Find best result for visualization (prefer higher confidence)
            best_text = "No text detected"
            best_confidence = 0.0
            
            for desc, threshold, text, details in all_threshold_results:
                # Calculate average confidence
                if details:
                    avg_confidence = sum(conf for _, conf in details) / len(details)
                    # If text is longer or same length but higher confidence
                    if (len(text) > len(best_text)) or (len(text) == len(best_text) and avg_confidence > best_confidence):
                        best_text = text
                        best_confidence = avg_confidence
            
            # Add the best OCR result to the detection image
            if best_text != "No text detected":
                cv2.putText(detection_image, f"OCR: {best_text}", (x1, y2+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"✗ Error processing plate #{plate_idx+1}: {e}")

# Save the detection visualization
cv2.imwrite('yolo_easyocr_threshold/all_detections_with_ocr.jpg', detection_image)

# Check if any plates were detected
if plates_detected == 0:
    print("\n✗ No license plates were detected in the image!")
    print("Possible issues:")
    print("1. The image doesn't contain clearly visible license plates")
    print("2. The YOLO model isn't configured correctly for this type of plate")
    print("3. The confidence threshold may be too high")
else:
    print(f"\n✓ Successfully processed {plates_detected} license plates")
    print("Check the 'yolo_easyocr_threshold' folder for results")