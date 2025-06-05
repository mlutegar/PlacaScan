import cv2
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime

class PlateValidationTool:
    def __init__(self, results_dir='video_analysis_results'):
        self.results_dir = results_dir
        self.current_plate_index = 0
        self.plates_data = []
        self.all_plates_data = []  # Keep original for reference
        self.validation_results = []
        self.filter_mode = "all"  # "all", "deduplicated", "high_quality_only"
        
        # Load data
        self.load_plates_data()
        
        # Create GUI
        self.setup_gui()
        
        # Load first plate
        if self.plates_data:
            self.load_current_plate()
    
    def load_plates_data(self):
        """Load all plates data from the batch processing results"""
        summary_file = os.path.join(self.results_dir, 'results', 'complete_batch_summary.json')
        
        if not os.path.exists(summary_file):
            messagebox.showerror("Error", f"Results file not found: {summary_file}")
            return
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
            self.all_plates_data = data.get('plates_data', [])
        
        # Apply initial filter (show all by default)
        self.apply_filter()
        
        print(f"Loaded {len(self.all_plates_data)} total plates")
        print(f"Displaying {len(self.plates_data)} plates after filtering")
        
        # Load existing validation results if they exist
        self.validation_file = os.path.join(self.results_dir, 'validation_results.json')
        if os.path.exists(self.validation_file):
            with open(self.validation_file, 'r') as f:
                self.validation_results = json.load(f)
            print(f"Loaded {len(self.validation_results)} existing validations")
        else:
            self.validation_results = []
    
    def apply_filter(self):
        """Apply current filter to plates data"""
        if self.filter_mode == "all":
            self.plates_data = self.all_plates_data.copy()
        elif self.filter_mode == "deduplicated":
            self.plates_data = self.deduplicate_plates()
        elif self.filter_mode == "high_quality_only":
            self.plates_data = self.filter_high_quality_plates()
        
        # Reset index if it's out of bounds
        if self.current_plate_index >= len(self.plates_data):
            self.current_plate_index = 0
    
    def deduplicate_plates(self):
        """Remove visually similar plates from the same video within a time window"""
        deduplicated = []
        seen_plates = {}  # video_name -> list of (timestamp, plate_data)
        
        time_window = 5.0  # seconds - consider plates within 5 seconds as potential duplicates
        similarity_threshold = 0.8  # minimum similarity to consider as duplicate
        
        for plate in self.all_plates_data:
            video_name = plate['video_name']
            timestamp = plate['timestamp']
            
            if video_name not in seen_plates:
                seen_plates[video_name] = []
            
            # Check if this plate is similar to any recent plate from the same video
            is_duplicate = False
            for prev_timestamp, prev_plate in seen_plates[video_name]:
                if abs(timestamp - prev_timestamp) <= time_window:
                    # Check if plates are similar (simple bbox area comparison)
                    similarity = self.calculate_plate_similarity(plate, prev_plate)
                    if similarity > similarity_threshold:
                        # Keep the one with higher YOLO confidence
                        if plate['confidence'] > prev_plate['confidence']:
                            # Replace the previous plate with this one
                            deduplicated.remove(prev_plate)
                            seen_plates[video_name].remove((prev_timestamp, prev_plate))
                        else:
                            is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(plate)
                seen_plates[video_name].append((timestamp, plate))
        
        print(f"Deduplication: {len(self.all_plates_data)} -> {len(deduplicated)} plates")
        return deduplicated
    
    def calculate_plate_similarity(self, plate1, plate2):
        """Calculate similarity between two plates based on bbox size and position"""
        bbox1 = plate1['bbox']
        bbox2 = plate2['bbox']
        
        # Calculate areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate overlap if bboxes are similar sized
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # Calculate center distance
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        max_dimension = max(bbox1[2] - bbox1[0], bbox1[3] - bbox1[1], 
                           bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])
        
        position_similarity = max(0, 1 - (distance / max_dimension)) if max_dimension > 0 else 0
        
        # Combined similarity score
        return (area_ratio * 0.6 + position_similarity * 0.4)
    
    def filter_high_quality_plates(self):
        """Filter plates keeping only those with high YOLO confidence and good OCR potential"""
        high_quality = []
        
        confidence_threshold = 0.7  # Only plates with high YOLO confidence
        min_bbox_area = 2000  # Minimum bounding box area (pixels)
        
        for plate in self.all_plates_data:
            # Check YOLO confidence
            if plate['confidence'] < confidence_threshold:
                continue
            
            # Check bbox size
            bbox = plate['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < min_bbox_area:
                continue
            
            # Check if any OCR method detected text
            has_text = False
            if 'ocr_results' in plate:
                for method, thresholds in plate['ocr_results'].items():
                    for threshold, result in thresholds.items():
                        if result.get('text', '').strip():
                            has_text = True
                            break
                    if has_text:
                        break
            
            if has_text:
                high_quality.append(plate)
        
        print(f"High quality filter: {len(self.all_plates_data)} -> {len(high_quality)} plates")
        return high_quality
    
    def setup_gui(self):
        """Create the validation interface"""
        self.root = tk.Tk()
        self.root.title("License Plate OCR Validation Tool")
        self.root.geometry("1200x800")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Left panel - Image display
        image_frame = ttk.LabelFrame(main_frame, text="Plate Images", padding="5")
        image_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Original plate image
        self.original_label = ttk.Label(image_frame, text="Original Plate")
        self.original_label.grid(row=0, column=0, padx=5)
        
        # Best processed image
        self.processed_label = ttk.Label(image_frame, text="Best Processed")
        self.processed_label.grid(row=0, column=1, padx=5)
        
        # Progress info
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="Plate 0 of 0")
        self.progress_label.grid(row=0, column=0)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        progress_frame.columnconfigure(1, weight=1)
        
        # Plate information
        info_frame = ttk.LabelFrame(main_frame, text="Plate Information", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=4, width=80)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(0, weight=1)
        
        # OCR Results section
        ocr_frame = ttk.LabelFrame(main_frame, text="OCR Results", padding="5")
        ocr_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Create treeview for OCR results
        columns = ('Method', 'Threshold', 'OCR Text', 'Confidence', 'Status')
        self.ocr_tree = ttk.Treeview(ocr_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.ocr_tree.heading(col, text=col)
            if col == 'OCR Text':
                self.ocr_tree.column(col, width=200)
            elif col == 'Method':
                self.ocr_tree.column(col, width=120)
            else:
                self.ocr_tree.column(col, width=100)
        
        self.ocr_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(ocr_frame, orient=tk.VERTICAL, command=self.ocr_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.ocr_tree.configure(yscrollcommand=scrollbar.set)
        
        ocr_frame.columnconfigure(0, weight=1)
        ocr_frame.rowconfigure(0, weight=1)
        
        # Validation section
        validation_frame = ttk.LabelFrame(main_frame, text="Validation", padding="5")
        validation_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Ground truth input
        ttk.Label(validation_frame, text="Correct License Plate Text:").grid(row=0, column=0, sticky=tk.W)
        self.ground_truth_entry = ttk.Entry(validation_frame, width=30, font=('Arial', 12))
        self.ground_truth_entry.grid(row=0, column=1, padx=(10, 0), sticky=(tk.W, tk.E))
        
        # Quality assessment
        ttk.Label(validation_frame, text="Image Quality:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.quality_var = tk.StringVar(value="Good")
        quality_frame = ttk.Frame(validation_frame)
        quality_frame.grid(row=1, column=1, padx=(10, 0), sticky=tk.W, pady=(10, 0))
        
        ttk.Radiobutton(quality_frame, text="Excellent", variable=self.quality_var, value="Excellent").grid(row=0, column=0)
        ttk.Radiobutton(quality_frame, text="Good", variable=self.quality_var, value="Good").grid(row=0, column=1, padx=(10, 0))
        ttk.Radiobutton(quality_frame, text="Poor", variable=self.quality_var, value="Poor").grid(row=0, column=2, padx=(10, 0))
        ttk.Radiobutton(quality_frame, text="Unreadable", variable=self.quality_var, value="Unreadable").grid(row=0, column=3, padx=(10, 0))
        
        # Notes
        ttk.Label(validation_frame, text="Notes:").grid(row=2, column=0, sticky=(tk.W, tk.N), pady=(10, 0))
        self.notes_text = tk.Text(validation_frame, height=3, width=50)
        self.notes_text.grid(row=2, column=1, padx=(10, 0), pady=(10, 0), sticky=(tk.W, tk.E))
        
        validation_frame.columnconfigure(1, weight=1)
        
        # Navigation and filtering buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Filter options
        filter_frame = ttk.LabelFrame(nav_frame, text="Filter Options", padding="5")
        filter_frame.grid(row=0, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(filter_frame, text="Show:").grid(row=0, column=0, padx=(0, 10))
        
        self.filter_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text=f"All Plates ({len(self.all_plates_data)})", 
                       variable=self.filter_var, value="all", 
                       command=self.change_filter).grid(row=0, column=1)
        ttk.Radiobutton(filter_frame, text="Deduplicated", 
                       variable=self.filter_var, value="deduplicated",
                       command=self.change_filter).grid(row=0, column=2, padx=(10, 0))
        ttk.Radiobutton(filter_frame, text="High Quality Only", 
                       variable=self.filter_var, value="high_quality_only",
                       command=self.change_filter).grid(row=0, column=3, padx=(10, 0))
        
        # Navigation buttons
        nav_buttons_frame = ttk.Frame(nav_frame)
        nav_buttons_frame.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(nav_buttons_frame, text="◀ Previous", command=self.previous_plate).grid(row=0, column=0)
        ttk.Button(nav_buttons_frame, text="Save & Next ▶", command=self.save_and_next).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(nav_buttons_frame, text="Skip", command=self.skip_plate).grid(row=0, column=2, padx=(10, 0))
        ttk.Button(nav_buttons_frame, text="Go to Plate #", command=self.go_to_plate).grid(row=0, column=3, padx=(20, 0))
        ttk.Button(nav_buttons_frame, text="Export Results", command=self.export_results).grid(row=0, column=4, padx=(10, 0))
        ttk.Button(nav_buttons_frame, text="Mark as Duplicate", command=self.mark_duplicate).grid(row=0, column=5, padx=(10, 0))
        
        # Configure main frame grid weights
        main_frame.rowconfigure(3, weight=1)
        
        # Bind keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.previous_plate())
        self.root.bind('<Right>', lambda e: self.save_and_next())
        self.root.bind('<Return>', lambda e: self.save_and_next())
        self.root.bind('<space>', lambda e: self.skip_plate())
        
        # Focus on ground truth entry
        self.ground_truth_entry.focus()
    
    def load_current_plate(self):
        """Load the current plate data into the interface"""
        if not self.plates_data or self.current_plate_index >= len(self.plates_data):
            return
        
        plate = self.plates_data[self.current_plate_index]
        
        # Update progress
        self.progress_label.config(text=f"Plate {self.current_plate_index + 1} of {len(self.plates_data)}")
        self.progress_bar.config(maximum=len(self.plates_data), value=self.current_plate_index + 1)
        
        # Load and display images
        self.load_images(plate)
        
        # Update plate information
        info_text = f"Video: {plate['video_name']}\n"
        info_text += f"Frame: {plate['frame_number']} ({plate['timestamp']:.2f}s)\n"
        info_text += f"YOLO Confidence: {plate['confidence']:.3f}\n"
        info_text += f"File: {plate['filename']}"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        
        # Clear and populate OCR results
        for item in self.ocr_tree.get_children():
            self.ocr_tree.delete(item)
        
        if 'ocr_results' in plate:
            self.populate_ocr_results(plate['ocr_results'])
        
        # Load existing validation if available
        existing_validation = self.get_existing_validation(plate['filename'])
        if existing_validation:
            self.ground_truth_entry.delete(0, tk.END)
            self.ground_truth_entry.insert(0, existing_validation.get('ground_truth', ''))
            self.quality_var.set(existing_validation.get('quality', 'Good'))
            self.notes_text.delete(1.0, tk.END)
            self.notes_text.insert(1.0, existing_validation.get('notes', ''))
        else:
            # Clear fields for new validation
            self.ground_truth_entry.delete(0, tk.END)
            self.quality_var.set('Good')
            self.notes_text.delete(1.0, tk.END)
        
        # Focus on ground truth entry
        self.ground_truth_entry.focus()
    
    def load_images(self, plate):
        """Load and display the original and best processed images"""
        try:
            # Load original cropped plate
            original_path = plate['crop_path']
            if os.path.exists(original_path):
                original_img = Image.open(original_path)
                original_img = self.resize_image_for_display(original_img, max_width=300)
                original_photo = ImageTk.PhotoImage(original_img)
                self.original_label.config(image=original_photo, text="")
                self.original_label.image = original_photo  # Keep a reference
            
            # Find and load the best processed image (highest confidence OCR result)
            best_method = self.find_best_ocr_method(plate.get('ocr_results', {}))
            if best_method:
                processed_path = f"{self.results_dir}/processed_plates/{plate['filename']}_{best_method.lower()}.jpg"
                if os.path.exists(processed_path):
                    processed_img = Image.open(processed_path)
                    processed_img = self.resize_image_for_display(processed_img, max_width=300)
                    processed_photo = ImageTk.PhotoImage(processed_img)
                    self.processed_label.config(image=processed_photo, text=f"Best: {best_method}")
                    self.processed_label.image = processed_photo
                else:
                    self.processed_label.config(image="", text="No processed image")
            else:
                self.processed_label.config(image="", text="No OCR results")
                
        except Exception as e:
            print(f"Error loading images: {e}")
    
    def resize_image_for_display(self, img, max_width=300, max_height=200):
        """Resize image for display while maintaining aspect ratio"""
        width, height = img.size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def find_best_ocr_method(self, ocr_results):
        """Find the preprocessing method with the highest confidence OCR result"""
        best_method = None
        best_confidence = 0
        
        for method, thresholds in ocr_results.items():
            for threshold, result in thresholds.items():
                if result['confidence'] > best_confidence:
                    best_confidence = result['confidence']
                    best_method = method
        
        return best_method
    
    def populate_ocr_results(self, ocr_results):
        """Populate the OCR results treeview"""
        for method, thresholds in ocr_results.items():
            for threshold, result in thresholds.items():
                text = result['text']
                confidence = f"{result['confidence']:.3f}"
                
                # Determine status based on text detection
                status = "✓ Detected" if text else "✗ No text"
                
                self.ocr_tree.insert('', 'end', values=(
                    method, threshold, text, confidence, status
                ))
    
    def change_filter(self):
        """Change the current filter and update display"""
        old_filter = self.filter_mode
        self.filter_mode = self.filter_var.get()
        
        if old_filter != self.filter_mode:
            self.apply_filter()
            
            if self.plates_data:
                self.load_current_plate()
            else:
                messagebox.showinfo("No Plates", "No plates match the current filter criteria.")
    
    def mark_duplicate(self):
        """Mark current plate as duplicate and skip"""
        if self.current_plate_index >= len(self.plates_data):
            return
        
        plate = self.plates_data[self.current_plate_index]
        
        validation_data = {
            'filename': plate['filename'],
            'video_name': plate['video_name'],
            'frame_number': plate['frame_number'],
            'timestamp': plate['timestamp'],
            'yolo_confidence': plate['confidence'],
            'ground_truth': '',
            'quality': 'Duplicate',
            'notes': 'Marked as duplicate during validation',
            'validation_date': datetime.now().isoformat(),
            'is_duplicate': True,
            'ocr_accuracies': {}
        }
        
        # Remove existing validation for this plate if it exists
        self.validation_results = [v for v in self.validation_results if v['filename'] != plate['filename']]
        
        # Add duplicate marker
        self.validation_results.append(validation_data)
        
        # Save to file
        with open(self.validation_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"Marked {plate['filename']} as duplicate")
        self.next_plate()
    
    def get_existing_validation(self, filename):
        """Check if validation already exists for this plate"""
        for validation in self.validation_results:
            if validation['filename'] == filename:
                return validation
        return None
    
    def save_validation(self):
        """Save the current validation"""
        if self.current_plate_index >= len(self.plates_data):
            return
        
        plate = self.plates_data[self.current_plate_index]
        
        # Calculate accuracy for each OCR method
        ground_truth = self.ground_truth_entry.get().strip().upper()
        ocr_accuracies = {}
        
        if ground_truth and 'ocr_results' in plate:
            for method, thresholds in plate['ocr_results'].items():
                method_accuracies = {}
                for threshold, result in thresholds.items():
                    ocr_text = result['text'].upper()
                    is_correct = (ocr_text == ground_truth)
                    method_accuracies[str(threshold)] = {
                        'ocr_text': result['text'],
                        'confidence': result['confidence'],
                        'is_correct': is_correct
                    }
                ocr_accuracies[method] = method_accuracies
        
        validation_data = {
            'filename': plate['filename'],
            'video_name': plate['video_name'],
            'frame_number': plate['frame_number'],
            'timestamp': plate['timestamp'],
            'yolo_confidence': plate['confidence'],
            'ground_truth': ground_truth,
            'quality': self.quality_var.get(),
            'notes': self.notes_text.get(1.0, tk.END).strip(),
            'validation_date': datetime.now().isoformat(),
            'ocr_accuracies': ocr_accuracies
        }
        
        # Remove existing validation for this plate if it exists
        self.validation_results = [v for v in self.validation_results if v['filename'] != plate['filename']]
        
        # Add new validation
        self.validation_results.append(validation_data)
        
        # Save to file
        with open(self.validation_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"Saved validation for {plate['filename']}")
    
    def save_and_next(self):
        """Save current validation and move to next plate"""
        self.save_validation()
        self.next_plate()
    
    def skip_plate(self):
        """Skip current plate without saving validation"""
        self.next_plate()
    
    def next_plate(self):
        """Move to the next plate"""
        if self.current_plate_index < len(self.plates_data) - 1:
            self.current_plate_index += 1
            self.load_current_plate()
        else:
            messagebox.showinfo("Complete", "All plates have been processed!")
    
    def previous_plate(self):
        """Move to the previous plate"""
        if self.current_plate_index > 0:
            self.current_plate_index -= 1
            self.load_current_plate()
    
    def go_to_plate(self):
        """Go to a specific plate number"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Go to Plate")
        dialog.geometry("300x100")
        
        ttk.Label(dialog, text="Plate number (1-{}):".format(len(self.plates_data))).pack(pady=10)
        
        entry = ttk.Entry(dialog)
        entry.pack(pady=5)
        entry.focus()
        
        def go():
            try:
                plate_num = int(entry.get())
                if 1 <= plate_num <= len(self.plates_data):
                    self.current_plate_index = plate_num - 1
                    self.load_current_plate()
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", f"Plate number must be between 1 and {len(self.plates_data)}")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
        
        ttk.Button(dialog, text="Go", command=go).pack(pady=10)
        entry.bind('<Return>', lambda e: go())
    
    def export_results(self):
        """Export validation results to CSV and detailed analysis"""
        if not self.validation_results:
            messagebox.showwarning("Warning", "No validation results to export")
            return
        
        # Create summary CSV
        summary_data = []
        detailed_data = []
        
        for validation in self.validation_results:
            # Summary row
            summary_row = {
                'filename': validation['filename'],
                'video_name': validation['video_name'],
                'ground_truth': validation['ground_truth'],
                'quality': validation['quality'],
                'yolo_confidence': validation['yolo_confidence']
            }
            
            # Add best OCR result for each method
            if 'ocr_accuracies' in validation:
                for method, thresholds in validation['ocr_accuracies'].items():
                    best_result = max(thresholds.values(), key=lambda x: x['confidence'], default={})
                    summary_row[f'{method}_best_text'] = best_result.get('ocr_text', '')
                    summary_row[f'{method}_best_confidence'] = best_result.get('confidence', 0)
                    summary_row[f'{method}_best_correct'] = best_result.get('is_correct', False)
            
            summary_data.append(summary_row)
            
            # Detailed rows for each method/threshold combination
            if 'ocr_accuracies' in validation:
                for method, thresholds in validation['ocr_accuracies'].items():
                    for threshold, result in thresholds.items():
                        detailed_row = {
                            'filename': validation['filename'],
                            'video_name': validation['video_name'],
                            'ground_truth': validation['ground_truth'],
                            'quality': validation['quality'],
                            'yolo_confidence': validation['yolo_confidence'],
                            'preprocessing_method': method,
                            'confidence_threshold': threshold,
                            'ocr_text': result['ocr_text'],
                            'ocr_confidence': result['confidence'],
                            'is_correct': result['is_correct']
                        }
                        detailed_data.append(detailed_row)
        
        # Save CSV files
        summary_df = pd.DataFrame(summary_data)
        detailed_df = pd.DataFrame(detailed_data)
        
        summary_csv = os.path.join(self.results_dir, 'validation_summary.csv')
        detailed_csv = os.path.join(self.results_dir, 'validation_detailed.csv')
        
        summary_df.to_csv(summary_csv, index=False)
        detailed_df.to_csv(detailed_csv, index=False)
        
        # Calculate accuracy statistics
        if not detailed_df.empty:
            accuracy_stats = detailed_df.groupby(['preprocessing_method', 'confidence_threshold'])['is_correct'].agg(['count', 'sum', 'mean']).reset_index()
            accuracy_stats.columns = ['preprocessing_method', 'confidence_threshold', 'total_tests', 'correct_predictions', 'accuracy']
            
            stats_csv = os.path.join(self.results_dir, 'accuracy_statistics.csv')
            accuracy_stats.to_csv(stats_csv, index=False)
            
            messagebox.showinfo("Export Complete", 
                               f"Results exported to:\n"
                               f"• {summary_csv}\n"
                               f"• {detailed_csv}\n"
                               f"• {stats_csv}\n\n"
                               f"Total validated plates: {len(summary_data)}")
        else:
            messagebox.showinfo("Export Complete", f"Summary exported to: {summary_csv}")
    
    def run(self):
        """Start the validation tool"""
        if not self.plates_data:
            messagebox.showerror("Error", "No plates data found. Please run the video processing first.")
            return
        
        print("\nLicense Plate Validation Tool")
        print("=============================")
        print("Keyboard shortcuts:")
        print("← Previous plate")
        print("→ or Enter: Save & Next")
        print("Space: Skip plate")
        print("\nInstructions:")
        print("1. Look at the original and processed plate images")
        print("2. Enter the correct license plate text (ground truth)")
        print("3. Select the image quality")
        print("4. Add any notes if needed")
        print("5. Press 'Save & Next' or Enter to continue")
        print("\nStarting validation...")
        
        self.root.mainloop()

# Usage
if __name__ == "__main__":
    # Initialize and run the validation tool
    validator = PlateValidationTool('video_analysis_results')
    validator.run()