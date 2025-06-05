from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime
import easyocr

class VideoPlateDetector:
    def __init__(self, model_path='placa-veicular-model.pt', output_dir='video_plates_output'):
        """
        Initialize the video plate detector
        
        Args:
            model_path: Path to YOLO model
            output_dir: Directory to save results
        """
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en', 'pt'])
        self.output_dir = output_dir
        
        # Create output directories
        self.setup_directories()
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.frame_skip = 5  # Process every 5th frame for efficiency
        
    def setup_directories(self):
        """Create necessary output directories"""
        directories = [
            self.output_dir,
            f"{self.output_dir}/cropped_plates",
            f"{self.output_dir}/processed_plates",
            f"{self.output_dir}/video_frames",
            f"{self.output_dir}/results"
        ]
        
        for dir_path in directories:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def detect_and_crop_plates(self, video_path, max_plates_per_video=50):
        """
        Process video to detect and crop license plates
        
        Args:
            video_path: Path to input video
            max_plates_per_video: Maximum number of plates to extract per video
            
        Returns:
            List of cropped plate information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        plates_detected = []
        frame_number = 0
        plates_count = 0
        
        # Video name for file naming
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        while cap.isOpened() and plates_count < max_plates_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_number % self.frame_skip != 0:
                frame_number += 1
                continue
            
            # Calculate timestamp
            timestamp = frame_number / fps
            
            # Run YOLO detection
            results = self.model(frame)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    continue
                
                for box_idx, box in enumerate(boxes):
                    confidence = float(box.conf[0])
                    
                    # Filter by confidence
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Extract coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Add padding
                    padding = 10
                    h, w = frame.shape[:2]
                    y1_pad = max(0, y1 - padding)
                    y2_pad = min(h, y2 + padding)
                    x1_pad = max(0, x1 - padding)
                    x2_pad = min(w, x2 + padding)
                    
                    # Crop the plate
                    cropped_plate = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                    
                    # Skip if crop is too small
                    if cropped_plate.shape[0] < 20 or cropped_plate.shape[1] < 50:
                        continue
                    
                    # Generate unique filename
                    plate_filename = f"{video_name}_frame{frame_number}_plate{box_idx}_{timestamp:.2f}s"
                    
                    # Save cropped plate
                    crop_path = f"{self.output_dir}/cropped_plates/{plate_filename}.jpg"
                    cv2.imwrite(crop_path, cropped_plate)
                    
                    # Save frame with detection for reference
                    frame_with_detection = frame.copy()
                    cv2.rectangle(frame_with_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_with_detection, f"Conf: {confidence:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    frame_path = f"{self.output_dir}/video_frames/{plate_filename}_frame.jpg"
                    cv2.imwrite(frame_path, frame_with_detection)
                    
                    # Store plate information
                    plate_info = {
                        'video_name': video_name,
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'crop_path': crop_path,
                        'frame_path': frame_path,
                        'filename': plate_filename
                    }
                    
                    plates_detected.append(plate_info)
                    plates_count += 1
                    
                    print(f"Plate {plates_count}: Frame {frame_number} ({timestamp:.2f}s) - Conf: {confidence:.2f}")
                    
                    if plates_count >= max_plates_per_video:
                        break
            
            frame_number += 1
            
            # Progress update
            if frame_number % (self.frame_skip * 10) == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Plates found: {plates_count}")
        
        cap.release()
        
        # Save detection results
        results_file = f"{self.output_dir}/results/{video_name}_detections.json"
        with open(results_file, 'w') as f:
            json.dump(plates_detected, f, indent=2)
        
        print(f"Completed: {len(plates_detected)} plates extracted from {video_name}")
        return plates_detected
    
    def apply_preprocessing_methods(self, plate_info):
        """
        Apply various preprocessing methods to a cropped plate
        
        Args:
            plate_info: Dictionary containing plate information
            
        Returns:
            Dictionary of preprocessed images
        """
        # Load the cropped plate
        cropped_plate = cv2.imread(plate_info['crop_path'])
        if cropped_plate is None:
            return {}
        
        filename = plate_info['filename']
        processed_images = {}
        
        # Original
        processed_images['Original'] = cropped_plate
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        processed_images['Grayscale'] = gray
        
        # Otsu thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images['Otsu'] = otsu
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_images['Adaptive'] = adaptive
        
        # Bilateral filter
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        processed_images['Bilateral'] = bilateral
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images['Sharpened'] = sharpened
        
        # Resize 2x
        h, w = gray.shape
        resized = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        processed_images['Resized2x'] = resized
        
        # Inverted
        inverted = cv2.bitwise_not(gray)
        processed_images['Inverted'] = inverted
        
        # Save all processed versions
        for method_name, img in processed_images.items():
            if method_name != 'Original':  # Skip original as it's already saved
                save_path = f"{self.output_dir}/processed_plates/{filename}_{method_name.lower()}.jpg"
                cv2.imwrite(save_path, img)
        
        return processed_images
    
    def run_ocr_analysis(self, plate_info, confidence_thresholds=[0.0, 0.2, 0.4, 0.6, 0.8]):
        """
        Run OCR analysis on all preprocessing methods
        
        Args:
            plate_info: Dictionary containing plate information
            confidence_thresholds: List of confidence thresholds to test
            
        Returns:
            Dictionary of OCR results
        """
        # Apply preprocessing
        processed_images = self.apply_preprocessing_methods(plate_info)
        
        ocr_results = {}
        filename = plate_info['filename']
        
        for method_name, img in processed_images.items():
            method_results = {}
            
            try:
                # Get raw OCR results
                raw_results = self.reader.readtext(img)
                
                # Process for each threshold
                for threshold in confidence_thresholds:
                    filtered_results = [res for res in raw_results if res[2] >= threshold]
                    
                    if filtered_results:
                        # Extract and clean text
                        texts = []
                        for bbox, text, score in filtered_results:
                            clean_text = ''.join(c for c in text if c.isalnum())
                            if clean_text:
                                texts.append((clean_text, score))
                        
                        if texts:
                            # Sort by confidence
                            texts.sort(key=lambda x: x[1], reverse=True)
                            combined_text = ''.join([t[0] for t in texts])
                            
                            method_results[threshold] = {
                                'text': combined_text,
                                'confidence': sum(t[1] for t in texts) / len(texts),
                                'details': texts
                            }
                
                ocr_results[method_name] = method_results
                
            except Exception as e:
                print(f"OCR error for {method_name}: {e}")
                ocr_results[method_name] = {}
        
        # Save OCR results
        results_file = f"{self.output_dir}/results/{filename}_ocr_results.json"
        with open(results_file, 'w') as f:
            json.dump(ocr_results, f, indent=2)
        
        return ocr_results
    
    def process_video_batch(self, video_directory, video_extensions=['.mp4', '.avi', '.mov', '.mkv']):
        """
        Process multiple videos in a directory
        
        Args:
            video_directory: Directory containing videos
            video_extensions: List of video file extensions to process
            
        Returns:
            Combined results from all videos
        """
        all_results = []
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend([f for f in os.listdir(video_directory) if f.lower().endswith(ext.lower())])
        
        print(f"Found {len(video_files)} video files to process")
        
        for video_file in video_files:
            video_path = os.path.join(video_directory, video_file)
            print(f"\nProcessing video: {video_file}")
            
            # Extract plates from video
            plates = self.detect_and_crop_plates(video_path)
            
            # Run OCR analysis on each plate
            for plate_info in plates:
                print(f"Running OCR analysis on {plate_info['filename']}")
                ocr_results = self.run_ocr_analysis(plate_info)
                plate_info['ocr_results'] = ocr_results
            
            all_results.extend(plates)
        
        # Save combined results
        summary_file = f"{self.output_dir}/results/batch_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nBatch processing complete!")
        print(f"Total plates extracted: {len(all_results)}")
        print(f"Results saved to: {summary_file}")
        
        return all_results

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = VideoPlateDetector(
        model_path='placa-veicular-model.pt',
        output_dir='video_analysis_results'
    )
    
    # Process all videos in the 'videos' folder
    videos_folder = 'videos'
    
    if not os.path.exists(videos_folder):
        print(f"Error: '{videos_folder}' folder not found!")
        print("Please make sure you have a 'videos' folder in the current directory.")
        exit(1)
    
    print(f"Processing all videos in '{videos_folder}' folder...")
    
    # Get list of video files
    video_files = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    for file in os.listdir(videos_folder):
        if any(file.endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if not video_files:
        print(f"No video files found in '{videos_folder}' folder!")
        print(f"Looking for files with extensions: {video_extensions}")
        exit(1)
    
    print(f"Found {len(video_files)} video files:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")
    
    # Process each video
    all_plates_data = []
    total_plates_extracted = 0
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(videos_folder, video_file)
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {video_file}")
        print(f"{'='*60}")
        
        # Extract plates from video
        plates = detector.detect_and_crop_plates(video_path, max_plates_per_video=30)
        
        if not plates:
            print(f"No plates detected in {video_file}")
            continue
        
        print(f"Extracted {len(plates)} plates from {video_file}")
        total_plates_extracted += len(plates)
        
        # Run OCR analysis on each plate
        print(f"Running OCR analysis on {len(plates)} plates...")
        
        for j, plate_info in enumerate(plates):
            print(f"  Processing plate {j+1}/{len(plates)}: {plate_info['filename']}")
            
            try:
                ocr_results = detector.run_ocr_analysis(plate_info)
                plate_info['ocr_results'] = ocr_results
                
                # Find best OCR result for logging
                best_text = ""
                best_confidence = 0
                best_method = ""
                
                for method, thresholds in ocr_results.items():
                    for threshold, result in thresholds.items():
                        if result['confidence'] > best_confidence:
                            best_text = result['text']
                            best_confidence = result['confidence']
                            best_method = f"{method} (threshold {threshold})"
                
                if best_text:
                    print(f"    Best OCR: '{best_text}' via {best_method} (conf: {best_confidence:.3f})")
                else:
                    print(f"    No text detected")
                    
            except Exception as e:
                print(f"    Error in OCR analysis: {e}")
                plate_info['ocr_results'] = {}
        
        all_plates_data.extend(plates)
        print(f"Completed {video_file}: {len(plates)} plates processed")
    
    # Save final summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Videos processed: {len(video_files)}")
    print(f"Total plates extracted: {total_plates_extracted}")
    print(f"Average plates per video: {total_plates_extracted/len(video_files):.1f}")
    
    # Save comprehensive results
    final_summary = {
        'processing_date': datetime.now().isoformat(),
        'videos_processed': len(video_files),
        'total_plates': total_plates_extracted,
        'video_files': video_files,
        'plates_data': all_plates_data
    }
    
    summary_file = f"{detector.output_dir}/results/complete_batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\nResults saved to: {summary_file}")
    print(f"Check '{detector.output_dir}' folder for all extracted plates and analysis results")
    
    # Generate quick statistics
    methods_tested = ['Original', 'Grayscale', 'Otsu', 'Adaptive', 'Bilateral', 'Sharpened', 'Resized2x', 'Inverted']
    thresholds_tested = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    print(f"\nDataset Statistics:")
    print(f"- Preprocessing methods tested: {len(methods_tested)}")
    print(f"- Confidence thresholds tested: {len(thresholds_tested)}")
    print(f"- Total combinations per plate: {len(methods_tested) * len(thresholds_tested)}")
    print(f"- Total OCR tests performed: {total_plates_extracted * len(methods_tested) * len(thresholds_tested)}")
    
    print(f"\nReady for academic analysis! 🎓📊")