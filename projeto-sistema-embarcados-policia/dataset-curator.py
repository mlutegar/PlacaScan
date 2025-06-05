import json
import os
import shutil
import pandas as pd
from datetime import datetime
import cv2
from collections import defaultdict

class DatasetCurator:
    def __init__(self, results_dir='video_analysis_results', curated_dir='curated_dataset'):
        self.results_dir = results_dir
        self.curated_dir = curated_dir
        self.validation_file = os.path.join(results_dir, 'validation_results.json')
        
        # Create curated dataset directory structure
        self.setup_curated_directories()
        
        # Load validation results
        self.load_validation_data()
    
    def setup_curated_directories(self):
        """Create directory structure for curated dataset"""
        directories = [
            self.curated_dir,
            f"{self.curated_dir}/cropped_plates",
            f"{self.curated_dir}/processed_plates", 
            f"{self.curated_dir}/results",
            f"{self.curated_dir}/metadata"
        ]
        
        for dir_path in directories:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        print(f"Created curated dataset structure in: {self.curated_dir}")
    
    def load_validation_data(self):
        """Load validation results from the analysis tool"""
        if not os.path.exists(self.validation_file):
            print(f"Error: Validation file not found: {self.validation_file}")
            print("Please run the validation tool first and validate some plates.")
            return None
        
        with open(self.validation_file, 'r') as f:
            self.validations = json.load(f)
        
        print(f"Loaded {len(self.validations)} validation records")
        return self.validations
    
    def filter_validated_plates(self, min_quality_levels=None, max_plates_per_video=3):
        """
        Filter plates based on validation criteria
        
        Args:
            min_quality_levels: List of acceptable quality levels (default: ['Excellent', 'Good'])
            max_plates_per_video: Maximum number of plates to keep per video
        """
        if min_quality_levels is None:
            min_quality_levels = ['Excellent', 'Good']
        
        # Filter criteria
        filtered_plates = []
        
        for validation in self.validations:
            # Skip if not annotated (no ground truth)
            if not validation.get('ground_truth', '').strip():
                print(f"Skipping {validation['filename']}: No ground truth annotation")
                continue
            
            # Skip duplicates
            if validation.get('is_duplicate', False):
                print(f"Skipping {validation['filename']}: Marked as duplicate")
                continue
            
            # Skip poor quality images
            quality = validation.get('quality', 'Unknown')
            if quality not in min_quality_levels:
                print(f"Skipping {validation['filename']}: Quality '{quality}' not in {min_quality_levels}")
                continue
            
            # Skip if marked as unreadable
            if quality == 'Unreadable':
                print(f"Skipping {validation['filename']}: Marked as unreadable")
                continue
            
            filtered_plates.append(validation)
        
        print(f"Filtered to {len(filtered_plates)} high-quality, annotated plates")
        
        # Group by video and limit per video
        video_groups = defaultdict(list)
        for plate in filtered_plates:
            video_groups[plate['video_name']].append(plate)
        
        # Select best plates per video
        curated_plates = []
        for video_name, plates in video_groups.items():
            # Sort by quality priority and YOLO confidence
            quality_priority = {'Excellent': 3, 'Good': 2, 'Poor': 1, 'Unreadable': 0}
            
            sorted_plates = sorted(plates, key=lambda x: (
                quality_priority.get(x.get('quality', 'Unknown'), 0),
                x.get('yolo_confidence', 0)
            ), reverse=True)
            
            # Take up to max_plates_per_video best plates
            selected = sorted_plates[:max_plates_per_video]
            curated_plates.extend(selected)
            
            print(f"Video '{video_name}': Selected {len(selected)} of {len(plates)} plates")
        
        print(f"Final curated dataset: {len(curated_plates)} plates from {len(video_groups)} videos")
        return curated_plates
    
    def calculate_quality_score(self, validation):
        """Calculate a quality score for ranking plates"""
        quality_scores = {'Excellent': 4, 'Good': 3, 'Poor': 2, 'Unreadable': 1}
        base_score = quality_scores.get(validation.get('quality', 'Poor'), 2)
        
        # Bonus for high YOLO confidence
        yolo_confidence = validation.get('yolo_confidence', 0)
        confidence_bonus = yolo_confidence * 2
        
        # Bonus for having successful OCR results
        ocr_bonus = 0
        if 'ocr_accuracies' in validation:
            # Check if any method got the correct result
            for method, thresholds in validation['ocr_accuracies'].items():
                for threshold, result in thresholds.items():
                    if result.get('is_correct', False):
                        ocr_bonus = 1
                        break
                if ocr_bonus > 0:
                    break
        
        total_score = base_score + confidence_bonus + ocr_bonus
        return total_score
    
    def copy_plate_files(self, curated_plates):
        """Copy selected plate files to curated dataset"""
        copied_files = []
        
        for validation in curated_plates:
            filename = validation['filename']
            
            # Copy original cropped plate
            original_src = os.path.join(self.results_dir, 'cropped_plates', f"{filename}.jpg")
            original_dst = os.path.join(self.curated_dir, 'cropped_plates', f"{filename}.jpg")
            
            if os.path.exists(original_src):
                shutil.copy2(original_src, original_dst)
                print(f"Copied: {filename}.jpg")
            else:
                print(f"Warning: Original file not found: {original_src}")
                continue
            
            # Copy all processed versions
            preprocessing_methods = ['grayscale', 'otsu', 'adaptive', 'bilateral', 'sharpened', 'resized2x', 'inverted']
            
            for method in preprocessing_methods:
                processed_src = os.path.join(self.results_dir, 'processed_plates', f"{filename}_{method}.jpg")
                processed_dst = os.path.join(self.curated_dir, 'processed_plates', f"{filename}_{method}.jpg")
                
                if os.path.exists(processed_src):
                    shutil.copy2(processed_src, processed_dst)
                else:
                    print(f"Warning: Processed file not found: {processed_src}")
            
            copied_files.append(filename)
        
        print(f"Successfully copied files for {len(copied_files)} plates")
        return copied_files
    
    def create_curated_metadata(self, curated_plates):
        """Create metadata files for the curated dataset"""
        
        # Create curated validation results
        curated_validation_file = os.path.join(self.curated_dir, 'results', 'curated_validation_results.json')
        with open(curated_validation_file, 'w') as f:
            json.dump(curated_plates, f, indent=2)
        
        # Create summary statistics
        summary_stats = self.generate_summary_statistics(curated_plates)
        
        # Save summary
        summary_file = os.path.join(self.curated_dir, 'metadata', 'dataset_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create CSV files for analysis
        self.create_analysis_csvs(curated_plates)
        
        print(f"Created metadata files in {self.curated_dir}/metadata/")
        return summary_stats
    
    def generate_summary_statistics(self, curated_plates):
        """Generate comprehensive statistics about the curated dataset"""
        
        # Basic statistics
        total_plates = len(curated_plates)
        unique_videos = len(set(p['video_name'] for p in curated_plates))
        
        # Quality distribution
        quality_dist = {}
        for plate in curated_plates:
            quality = plate.get('quality', 'Unknown')
            quality_dist[quality] = quality_dist.get(quality, 0) + 1
        
        # YOLO confidence statistics
        yolo_confidences = [p.get('yolo_confidence', 0) for p in curated_plates]
        yolo_stats = {
            'mean': sum(yolo_confidences) / len(yolo_confidences) if yolo_confidences else 0,
            'min': min(yolo_confidences) if yolo_confidences else 0,
            'max': max(yolo_confidences) if yolo_confidences else 0,
            'median': sorted(yolo_confidences)[len(yolo_confidences)//2] if yolo_confidences else 0
        }
        
        # Calculate OCR accuracy by method
        method_accuracies = self.calculate_method_accuracies(curated_plates)
        
        # Video distribution
        video_dist = {}
        for plate in curated_plates:
            video = plate['video_name']
            video_dist[video] = video_dist.get(video, 0) + 1
        
        summary = {
            'creation_date': datetime.now().isoformat(),
            'total_plates': total_plates,
            'unique_videos': unique_videos,
            'quality_distribution': quality_dist,
            'yolo_confidence_stats': yolo_stats,
            'method_accuracies': method_accuracies,
            'video_distribution': video_dist,
            'preprocessing_methods': ['Original', 'Grayscale', 'Otsu', 'Adaptive', 'Bilateral', 'Sharpened', 'Resized2x', 'Inverted'],
            'confidence_thresholds': [0.0, 0.2, 0.4, 0.6, 0.8]
        }
        
        return summary
    
    def calculate_method_accuracies(self, curated_plates):
        """Calculate accuracy statistics for each preprocessing method"""
        method_stats = {}
        
        for plate in curated_plates:
            if 'ocr_accuracies' not in plate:
                continue
            
            for method, thresholds in plate['ocr_accuracies'].items():
                if method not in method_stats:
                    method_stats[method] = {'total': 0, 'correct': 0, 'by_threshold': {}}
                
                for threshold, result in thresholds.items():
                    # Overall method stats
                    method_stats[method]['total'] += 1
                    if result.get('is_correct', False):
                        method_stats[method]['correct'] += 1
                    
                    # By threshold stats
                    if threshold not in method_stats[method]['by_threshold']:
                        method_stats[method]['by_threshold'][threshold] = {'total': 0, 'correct': 0}
                    
                    method_stats[method]['by_threshold'][threshold]['total'] += 1
                    if result.get('is_correct', False):
                        method_stats[method]['by_threshold'][threshold]['correct'] += 1
        
        # Calculate accuracy percentages
        for method in method_stats:
            total = method_stats[method]['total']
            correct = method_stats[method]['correct']
            method_stats[method]['accuracy'] = (correct / total * 100) if total > 0 else 0
            
            for threshold in method_stats[method]['by_threshold']:
                thresh_total = method_stats[method]['by_threshold'][threshold]['total']
                thresh_correct = method_stats[method]['by_threshold'][threshold]['correct']
                method_stats[method]['by_threshold'][threshold]['accuracy'] = (thresh_correct / thresh_total * 100) if thresh_total > 0 else 0
        
        return method_stats
    
    def create_analysis_csvs(self, curated_plates):
        """Create CSV files ready for statistical analysis"""
        
        # Detailed results for each method/threshold combination
        detailed_data = []
        summary_data = []
        
        for plate in curated_plates:
            # Basic plate info
            base_info = {
                'filename': plate['filename'],
                'video_name': plate['video_name'],
                'ground_truth': plate['ground_truth'],
                'quality': plate['quality'],
                'yolo_confidence': plate['yolo_confidence']
            }
            
            summary_data.append(base_info.copy())
            
            # Add OCR results
            if 'ocr_accuracies' in plate:
                for method, thresholds in plate['ocr_accuracies'].items():
                    for threshold, result in thresholds.items():
                        detailed_row = base_info.copy()
                        detailed_row.update({
                            'preprocessing_method': method,
                            'confidence_threshold': float(threshold),
                            'ocr_text': result['ocr_text'],
                            'ocr_confidence': result['confidence'],
                            'is_correct': result['is_correct']
                        })
                        detailed_data.append(detailed_row)
        
        # Save CSV files
        detailed_df = pd.DataFrame(detailed_data)
        summary_df = pd.DataFrame(summary_data)
        
        detailed_csv = os.path.join(self.curated_dir, 'results', 'detailed_results.csv')
        summary_csv = os.path.join(self.curated_dir, 'results', 'summary_results.csv')
        
        detailed_df.to_csv(detailed_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        
        # Create accuracy summary by method
        if not detailed_df.empty:
            accuracy_summary = detailed_df.groupby(['preprocessing_method', 'confidence_threshold']).agg({
                'is_correct': ['count', 'sum', 'mean'],
                'ocr_confidence': 'mean'
            }).round(4)
            
            accuracy_summary.columns = ['total_tests', 'correct_predictions', 'accuracy', 'avg_ocr_confidence']
            accuracy_summary = accuracy_summary.reset_index()
            
            accuracy_csv = os.path.join(self.curated_dir, 'results', 'accuracy_by_method.csv')
            accuracy_summary.to_csv(accuracy_csv, index=False)
            
            print(f"Created analysis CSV files:")
            print(f"  • {detailed_csv}")
            print(f"  • {summary_csv}")
            print(f"  • {accuracy_csv}")
    
    def create_curated_dataset(self, min_quality_levels=None, max_plates_per_video=3, 
                              include_poor_quality=False):
        """
        Main function to create the curated dataset
        
        Args:
            min_quality_levels: Quality levels to include (default: ['Excellent', 'Good'])
            max_plates_per_video: Maximum plates per video (default: 3)
            include_poor_quality: Whether to include 'Poor' quality images (default: False)
        """
        
        if self.validations is None:
            return None
        
        if min_quality_levels is None:
            if include_poor_quality:
                min_quality_levels = ['Excellent', 'Good', 'Poor']
            else:
                min_quality_levels = ['Excellent', 'Good']
        
        print(f"\nCreating curated dataset...")
        print(f"Quality levels: {min_quality_levels}")
        print(f"Max plates per video: {max_plates_per_video}")
        print("="*50)
        
        # Step 1: Filter plates based on validation criteria
        curated_plates = self.filter_validated_plates(min_quality_levels, max_plates_per_video)
        
        if not curated_plates:
            print("No plates met the curation criteria!")
            return None
        
        # Step 2: Copy files to curated dataset
        print("\nCopying files...")
        copied_files = self.copy_plate_files(curated_plates)
        
        # Step 3: Create metadata and analysis files
        print("\nGenerating metadata...")
        summary_stats = self.create_curated_metadata(curated_plates)
        
        # Step 4: Print summary
        print("\n" + "="*50)
        print("CURATED DATASET SUMMARY")
        print("="*50)
        print(f"Original dataset: {len(self.validations)} validated plates")
        print(f"Curated dataset: {len(curated_plates)} plates")
        print(f"Videos represented: {summary_stats['unique_videos']}")
        print(f"Quality distribution: {summary_stats['quality_distribution']}")
        print(f"Average YOLO confidence: {summary_stats['yolo_confidence_stats']['mean']:.3f}")
        
        print(f"\nMethod accuracies (overall):")
        for method, stats in summary_stats['method_accuracies'].items():
            print(f"  {method}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
        
        print(f"\nDataset location: {self.curated_dir}/")
        print(f"Ready for academic analysis! 🎓📊")
        
        return curated_plates
    
    def print_curation_options(self):
        """Print available curation options"""
        if self.validations is None:
            return
        
        # Analyze current validations
        quality_counts = {}
        annotated_count = 0
        duplicate_count = 0
        video_counts = {}
        
        for validation in self.validations:
            quality = validation.get('quality', 'Unknown')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            if validation.get('ground_truth', '').strip():
                annotated_count += 1
            
            if validation.get('is_duplicate', False):
                duplicate_count += 1
            
            video = validation['video_name']
            video_counts[video] = video_counts.get(video, 0) + 1
        
        print("\nCURRENT VALIDATION STATUS")
        print("="*40)
        print(f"Total validated: {len(self.validations)}")
        print(f"Annotated (have ground truth): {annotated_count}")
        print(f"Marked as duplicates: {duplicate_count}")
        print(f"\nQuality distribution:")
        for quality, count in sorted(quality_counts.items()):
            print(f"  {quality}: {count}")
        
        print(f"\nPlates per video:")
        for video, count in sorted(video_counts.items()):
            print(f"  {video}: {count}")
        
        print(f"\nRECOMMENDED CURATION OPTIONS:")
        print(f"• Conservative (Excellent only): ~{quality_counts.get('Excellent', 0)} plates")
        print(f"• Balanced (Excellent + Good): ~{quality_counts.get('Excellent', 0) + quality_counts.get('Good', 0)} plates")
        print(f"• Inclusive (Excellent + Good + Poor): ~{annotated_count - duplicate_count} plates")

# Usage example
if __name__ == "__main__":
    # Initialize curator
    curator = DatasetCurator('video_analysis_results', 'curated_dataset')
    
    # Show current validation status and options
    curator.print_curation_options()
    
    # Create curated dataset with different options:
    
    # Option 1: Conservative (only Excellent quality)
    # curated_plates = curator.create_curated_dataset(
    #     min_quality_levels=['Excellent'],
    #     max_plates_per_video=2
    # )
    
    # Option 2: Balanced (Excellent + Good quality) - RECOMMENDED
    curated_plates = curator.create_curated_dataset(
        min_quality_levels=['Excellent', 'Good'],
        max_plates_per_video=3
    )
    
    # Option 3: Inclusive (include Poor quality if needed)
    # curated_plates = curator.create_curated_dataset(
    #     min_quality_levels=['Excellent', 'Good', 'Poor'],
    #     max_plates_per_video=5
    # )