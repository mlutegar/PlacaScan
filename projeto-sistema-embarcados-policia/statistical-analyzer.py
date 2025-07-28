import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import json
import os
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class LicensePlateStatisticalAnalyzer:
    def __init__(self, curated_dir='curated_dataset'):
        self.curated_dir = curated_dir
        self.results_dir = os.path.join(curated_dir, 'results')
        
        # Load data
        self.load_data()
        
        # Brazilian license plate patterns
        self.old_pattern = re.compile(r'^[A-Z]{3}[0-9]{4}$')  # ABC1234
        self.new_pattern = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')  # ABC1A23 (Mercosul)
        
        # Setup plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load the curated dataset results"""
        detailed_file = os.path.join(self.results_dir, 'detailed_results.csv')
        summary_file = os.path.join(self.results_dir, 'summary_results.csv')
        
        if not os.path.exists(detailed_file):
            raise FileNotFoundError(f"Detailed results file not found: {detailed_file}")
        
        self.detailed_df = pd.read_csv(detailed_file)
        self.summary_df = pd.read_csv(summary_file) if os.path.exists(summary_file) else None
        
        print(f"Loaded {len(self.detailed_df)} OCR test results")
        print(f"Unique plates: {self.detailed_df['filename'].nunique()}")
        print(f"Preprocessing methods: {self.detailed_df['preprocessing_method'].nunique()}")
        print(f"Confidence thresholds: {sorted(self.detailed_df['confidence_threshold'].unique())}")
        
    def filter_valid_plates(self):
        """Filter to only include valid 7-character Brazilian license plates"""
        # Filter ground truth to 7-character plates
        def is_valid_plate(plate_text):
            if not isinstance(plate_text, str) or len(plate_text) != 7:
                return False
            return bool(self.old_pattern.match(plate_text) or self.new_pattern.match(plate_text))
        
        # Apply filter
        valid_mask = self.detailed_df['ground_truth'].apply(is_valid_plate)
        self.valid_plates_df = self.detailed_df[valid_mask].copy()
        
        print(f"\nFiltered to valid 7-character plates:")
        print(f"Valid plates: {len(self.valid_plates_df)} tests ({len(self.valid_plates_df)/len(self.detailed_df)*100:.1f}%)")
        print(f"Unique valid plates: {self.valid_plates_df['filename'].nunique()}")
        
        # Show ground truth distribution
        unique_plates = self.valid_plates_df.groupby('filename')['ground_truth'].first()
        old_format = sum(1 for plate in unique_plates if self.old_pattern.match(plate))
        new_format = sum(1 for plate in unique_plates if self.new_pattern.match(plate))
        
        print(f"Plate formats:")
        print(f"  Old format (ABC1234): {old_format}")
        print(f"  New format (ABC1A23): {new_format}")
        
        return self.valid_plates_df
    
    def calculate_character_accuracy(self):
        """Calculate character-level accuracy metrics"""
        char_results = []
        
        for _, row in self.valid_plates_df.iterrows():
            ground_truth = row['ground_truth']
            ocr_text = row['ocr_text']
            
            # Pad or truncate OCR text to 7 characters for comparison
            if len(ocr_text) < 7:
                ocr_text = ocr_text.ljust(7, ' ')  # Pad with spaces
            elif len(ocr_text) > 7:
                ocr_text = ocr_text[:7]  # Truncate
            
            # Character-by-character comparison
            char_matches = []
            for i in range(7):
                gt_char = ground_truth[i] if i < len(ground_truth) else ' '
                ocr_char = ocr_text[i] if i < len(ocr_text) else ' '
                char_matches.append(gt_char == ocr_char)
            
            char_result = {
                'filename': row['filename'],
                'preprocessing_method': row['preprocessing_method'],
                'confidence_threshold': row['confidence_threshold'],
                'ground_truth': ground_truth,
                'ocr_text': row['ocr_text'],
                'ocr_text_padded': ocr_text,
                'full_plate_correct': row['is_correct'],
                'characters_correct': sum(char_matches),
                'character_accuracy': sum(char_matches) / 7,
                'char_0_correct': char_matches[0],  # Position-specific accuracy
                'char_1_correct': char_matches[1],
                'char_2_correct': char_matches[2],
                'char_3_correct': char_matches[3],
                'char_4_correct': char_matches[4],
                'char_5_correct': char_matches[5],
                'char_6_correct': char_matches[6],
                'ocr_confidence': row['ocr_confidence'],
                'quality': row['quality']
            }
            char_results.append(char_result)
        
        self.char_df = pd.DataFrame(char_results)
        print(f"\nCharacter-level analysis completed for {len(self.char_df)} tests")
        return self.char_df
    
    def analyze_threshold_impact(self):
        """Compare performance with and without confidence thresholds"""
        
        # Group by method and compare thresholds
        threshold_analysis = []
        
        for method in self.valid_plates_df['preprocessing_method'].unique():
            method_data = self.valid_plates_df[self.valid_plates_df['preprocessing_method'] == method]
            
            for threshold in sorted(method_data['confidence_threshold'].unique()):
                thresh_data = method_data[method_data['confidence_threshold'] == threshold]
                
                # Calculate metrics
                total_tests = len(thresh_data)
                full_plate_accuracy = thresh_data['is_correct'].mean()
                avg_char_accuracy = self.char_df[
                    (self.char_df['preprocessing_method'] == method) & 
                    (self.char_df['confidence_threshold'] == threshold)
                ]['character_accuracy'].mean()
                
                avg_confidence = thresh_data['ocr_confidence'].mean()
                
                # Count detection rate (non-empty OCR results)
                detection_rate = (thresh_data['ocr_text'].str.len() > 0).mean()
                
                threshold_analysis.append({
                    'preprocessing_method': method,
                    'confidence_threshold': threshold,
                    'total_tests': total_tests,
                    'full_plate_accuracy': full_plate_accuracy,
                    'character_accuracy': avg_char_accuracy,
                    'detection_rate': detection_rate,
                    'avg_ocr_confidence': avg_confidence,
                    'threshold_category': 'No Threshold' if threshold == 0.0 else 'With Threshold'
                })
        
        self.threshold_df = pd.DataFrame(threshold_analysis)
        return self.threshold_df
    
    def calculate_false_positives(self):
        """Calculate false positive rates and analyze incorrect predictions"""
        fp_analysis = []
        
        for _, row in self.char_df.iterrows():
            if not row['full_plate_correct'] and len(row['ocr_text']) > 0:
                # This is a false positive (detected something but wrong)
                fp_type = 'false_positive'
            elif row['full_plate_correct']:
                fp_type = 'true_positive'
            elif len(row['ocr_text']) == 0:
                fp_type = 'false_negative'  # Should have detected but didn't
            else:
                fp_type = 'other'
            
            fp_analysis.append({
                'filename': row['filename'],
                'preprocessing_method': row['preprocessing_method'],
                'confidence_threshold': row['confidence_threshold'],
                'prediction_type': fp_type,
                'characters_correct': row['characters_correct'],
                'character_accuracy': row['character_accuracy'],
                'ocr_confidence': row['ocr_confidence']
            })
        
        self.fp_df = pd.DataFrame(fp_analysis)
        
        # Calculate false positive rates by method and threshold
        fp_rates = self.fp_df.groupby(['preprocessing_method', 'confidence_threshold', 'prediction_type']).size().unstack(fill_value=0)
        
        if 'false_positive' in fp_rates.columns:
            fp_rates['false_positive_rate'] = fp_rates['false_positive'] / (fp_rates.sum(axis=1))
        else:
            fp_rates['false_positive_rate'] = 0
        
        self.fp_rates = fp_rates
        return fp_rates
    
    def statistical_significance_tests(self):
        """Perform statistical significance tests"""
        results = {}
        
        # 1. Compare threshold vs no-threshold performance
        no_thresh = self.threshold_df[self.threshold_df['confidence_threshold'] == 0.0]
        with_thresh = self.threshold_df[self.threshold_df['confidence_threshold'] > 0.0]
        
        # T-test for full plate accuracy
        if len(no_thresh) > 0 and len(with_thresh) > 0:
            t_stat, p_val = ttest_ind(no_thresh['full_plate_accuracy'], with_thresh['full_plate_accuracy'])
            results['threshold_comparison'] = {
                'test': 'Independent t-test',
                'metric': 'Full plate accuracy',
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'no_threshold_mean': no_thresh['full_plate_accuracy'].mean(),
                'with_threshold_mean': with_thresh['full_plate_accuracy'].mean()
            }
        
        # 2. Compare preprocessing methods (ANOVA)
        methods_accuracy = []
        method_names = []
        
        for method in self.threshold_df['preprocessing_method'].unique():
            method_acc = self.threshold_df[self.threshold_df['preprocessing_method'] == method]['full_plate_accuracy']
            if len(method_acc) > 0:
                methods_accuracy.append(method_acc.values)
                method_names.append(method)
        
        if len(methods_accuracy) > 2:
            f_stat, p_val = stats.f_oneway(*methods_accuracy)
            results['methods_comparison'] = {
                'test': 'One-way ANOVA',
                'metric': 'Full plate accuracy across methods',
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'methods_tested': method_names
            }
        
        # 3. Character position analysis (Chi-square test)
        char_positions = ['char_0_correct', 'char_1_correct', 'char_2_correct', 
                         'char_3_correct', 'char_4_correct', 'char_5_correct', 'char_6_correct']
        
        position_accuracy = []
        for pos in char_positions:
            if pos in self.char_df.columns:
                position_accuracy.append(self.char_df[pos].mean())
        
        if len(position_accuracy) == 7:
            # Test if character positions have equal accuracy
            observed = [self.char_df[pos].sum() for pos in char_positions]
            total = len(self.char_df)
            expected = [total * (sum(observed) / (total * 7))] * 7
            
            chi2_stat, p_val = stats.chisquare(observed, expected)
            results['character_position'] = {
                'test': 'Chi-square goodness of fit',
                'metric': 'Character position accuracy equality',
                'chi2_statistic': chi2_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'position_accuracies': dict(zip(range(7), position_accuracy))
            }
        
        self.significance_results = results
        return results

    def create_individual_visualizations(self):
        """Create individual PNG files for each visualization"""

        # Create plots directory
        plots_dir = os.path.join(self.curated_dir, 'individual_plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # 1. Method comparison by accuracy
        plt.figure(figsize=(10, 6))
        method_summary = self.threshold_df.groupby('preprocessing_method')['full_plate_accuracy'].mean().sort_values(
            ascending=False)
        bars = plt.bar(range(len(method_summary)), method_summary.values)
        plt.title('Full Plate Accuracy by Preprocessing Method', fontsize=14, fontweight='bold')
        plt.xlabel('Preprocessing Method')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(method_summary)), method_summary.index, rotation=45, ha='right')
        plt.ylim(0, 1)

        for bar, val in zip(bars, method_summary.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '01_method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Threshold impact comparison
        plt.figure(figsize=(8, 6))
        threshold_comparison = self.threshold_df.groupby(['threshold_category'])['full_plate_accuracy'].mean()
        bars = plt.bar(threshold_comparison.index, threshold_comparison.values, color=['skyblue', 'orange'])
        plt.title('Threshold vs No-Threshold Performance', fontsize=14, fontweight='bold')
        plt.ylabel('Full Plate Accuracy')
        plt.ylim(0, 1)

        for bar, val in zip(bars, threshold_comparison.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '02_threshold_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Character vs Full Plate Accuracy
        plt.figure(figsize=(12, 6))
        char_vs_full = self.char_df.groupby('preprocessing_method')[['character_accuracy', 'full_plate_correct']].mean()
        x = np.arange(len(char_vs_full))
        width = 0.35

        plt.bar(x - width / 2, char_vs_full['character_accuracy'], width, label='Character Accuracy', alpha=0.8)
        plt.bar(x + width / 2, char_vs_full['full_plate_correct'], width, label='Full Plate Accuracy', alpha=0.8)

        plt.title('Character vs Full Plate Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Preprocessing Method')
        plt.ylabel('Accuracy')
        plt.xticks(x, char_vs_full.index, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '03_character_vs_full_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Character position accuracy
        plt.figure(figsize=(10, 6))
        char_positions = ['char_0_correct', 'char_1_correct', 'char_2_correct',
                          'char_3_correct', 'char_4_correct', 'char_5_correct', 'char_6_correct']

        position_accuracy = [self.char_df[pos].mean() for pos in char_positions if pos in self.char_df.columns]
        position_labels = ['Pos 0', 'Pos 1', 'Pos 2', 'Pos 3', 'Pos 4', 'Pos 5', 'Pos 6']

        bars = plt.bar(range(len(position_accuracy)), position_accuracy, color='lightcoral')
        plt.title('Accuracy by Character Position', fontsize=14, fontweight='bold')
        plt.xlabel('Character Position')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(position_accuracy)), position_labels[:len(position_accuracy)])
        plt.ylim(0, 1)

        for bar, val in zip(bars, position_accuracy):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '04_character_position_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. False Positive Analysis
        plt.figure(figsize=(8, 8))
        fp_summary = self.fp_df['prediction_type'].value_counts()
        colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightblue']
        plt.pie(fp_summary.values, labels=fp_summary.index, autopct='%1.1f%%', colors=colors[:len(fp_summary)])
        plt.title('Prediction Type Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '05_prediction_type_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Confidence vs Accuracy scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(self.char_df['ocr_confidence'], self.char_df['character_accuracy'], alpha=0.6)
        plt.xlabel('OCR Confidence')
        plt.ylabel('Character Accuracy')
        plt.title('OCR Confidence vs Character Accuracy', fontsize=14, fontweight='bold')

        correlation = np.corrcoef(self.char_df['ocr_confidence'], self.char_df['character_accuracy'])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '06_confidence_vs_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Method performance heatmap
        plt.figure(figsize=(12, 8))
        heatmap_data = self.threshold_df.pivot_table(values='full_plate_accuracy',
                                                     index='preprocessing_method',
                                                     columns='confidence_threshold',
                                                     fill_value=0)
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    cbar_kws={'label': 'Accuracy'})
        plt.title('Accuracy Heatmap: Method vs Threshold', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Preprocessing Method')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '07_method_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 8. Detection Rate vs Threshold
        plt.figure(figsize=(10, 6))
        detection_by_threshold = self.threshold_df.groupby('confidence_threshold')['detection_rate'].mean()
        plt.plot(detection_by_threshold.index, detection_by_threshold.values, marker='o', linewidth=2, markersize=8)
        plt.title('Detection Rate vs Confidence Threshold', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Detection Rate')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '08_detection_rate_vs_threshold.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 9. Quality vs Performance
        plt.figure(figsize=(10, 6))
        quality_performance = self.char_df.groupby('quality')[['character_accuracy', 'full_plate_correct']].mean()
        x = np.arange(len(quality_performance))
        width = 0.35

        plt.bar(x - width / 2, quality_performance['character_accuracy'], width, label='Character Accuracy', alpha=0.8)
        plt.bar(x + width / 2, quality_performance['full_plate_correct'], width, label='Full Plate Accuracy', alpha=0.8)

        plt.title('Performance by Image Quality', fontsize=14, fontweight='bold')
        plt.xlabel('Image Quality')
        plt.ylabel('Accuracy')
        plt.xticks(x, quality_performance.index)
        plt.legend()
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '09_quality_vs_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 10. Character accuracy distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.char_df['character_accuracy'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.char_df['character_accuracy'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {self.char_df["character_accuracy"].mean():.3f}')
        plt.title('Character Accuracy Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Character Accuracy')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '10_character_accuracy_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 11. Top performing combinations
        plt.figure(figsize=(12, 6))
        top_combinations = self.threshold_df.nlargest(10, 'full_plate_accuracy')
        combination_labels = [f"{row['preprocessing_method']}\n(T:{row['confidence_threshold']})"
                              for _, row in top_combinations.iterrows()]

        bars = plt.bar(range(len(top_combinations)), top_combinations['full_plate_accuracy'])
        plt.title('Top 10 Method-Threshold Combinations', fontsize=14, fontweight='bold')
        plt.xlabel('Method-Threshold Combination')
        plt.ylabel('Full Plate Accuracy')
        plt.xticks(range(len(top_combinations)), combination_labels, rotation=45, ha='right')
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '11_top_combinations.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 12. Statistical significance summary
        plt.figure(figsize=(10, 8))
        plt.axis('off')

        sig_text = "Statistical Significance Tests\n" + "=" * 30 + "\n\n"

        if hasattr(self, 'significance_results'):
            for test_name, result in self.significance_results.items():
                sig_text += f"{test_name.replace('_', ' ').title()}:\n"
                sig_text += f"  Test: {result['test']}\n"
                sig_text += f"  P-value: {result['p_value']:.4f}\n"
                sig_text += f"  Significant: {'Yes' if result['significant'] else 'No'}\n\n"

        plt.text(0.1, 0.9, sig_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '12_statistical_significance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Individual plots saved to: {plots_dir}")
        print("Generated 12 individual PNG files:")
        for i in range(1, 13):
            print(f"  • Plot {i:02d}: {plots_dir}")
    
    def generate_academic_report(self):
        """Generate a comprehensive academic report"""
        
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Dataset: {self.curated_dir}")
        report.append("")
        
        # Dataset Summary
        report.append("1. DATASET SUMMARY")
        report.append("-" * 20)
        report.append(f"Total OCR tests: {len(self.detailed_df):,}")
        report.append(f"Valid 7-character plates: {len(self.valid_plates_df):,}")
        report.append(f"Unique plates: {self.valid_plates_df['filename'].nunique()}")
        report.append(f"Preprocessing methods: {self.valid_plates_df['preprocessing_method'].nunique()}")
        report.append(f"Confidence thresholds tested: {sorted(self.valid_plates_df['confidence_threshold'].unique())}")
        report.append("")
        
        # Performance Overview
        report.append("2. PERFORMANCE OVERVIEW")
        report.append("-" * 25)
        overall_full_accuracy = self.char_df['full_plate_correct'].mean()
        overall_char_accuracy = self.char_df['character_accuracy'].mean()
        
        report.append(f"Overall full plate accuracy: {overall_full_accuracy:.3f} ({overall_full_accuracy*100:.1f}%)")
        report.append(f"Overall character accuracy: {overall_char_accuracy:.3f} ({overall_char_accuracy*100:.1f}%)")
        report.append("")
        
        # Best Performing Methods
        report.append("3. BEST PERFORMING METHODS")
        report.append("-" * 30)
        
        best_methods = self.threshold_df.nlargest(5, 'full_plate_accuracy')
        report.append("Top 5 Method-Threshold Combinations:")
        for i, (_, row) in enumerate(best_methods.iterrows(), 1):
            report.append(f"  {i}. {row['preprocessing_method']} (threshold {row['confidence_threshold']:.1f}): "
                         f"{row['full_plate_accuracy']:.3f} accuracy")
        report.append("")
        
        # Threshold Analysis
        report.append("4. THRESHOLD IMPACT ANALYSIS")
        report.append("-" * 32)
        
        no_thresh_acc = self.threshold_df[self.threshold_df['confidence_threshold'] == 0.0]['full_plate_accuracy'].mean()
        with_thresh_acc = self.threshold_df[self.threshold_df['confidence_threshold'] > 0.0]['full_plate_accuracy'].mean()
        
        report.append(f"No threshold (0.0): {no_thresh_acc:.3f} accuracy")
        report.append(f"With thresholds (>0.0): {with_thresh_acc:.3f} accuracy")
        report.append(f"Improvement: {((with_thresh_acc - no_thresh_acc) / no_thresh_acc * 100):+.1f}%")
        
        if hasattr(self, 'significance_results') and 'threshold_comparison' in self.significance_results:
            thresh_sig = self.significance_results['threshold_comparison']
            report.append(f"Statistical significance: p = {thresh_sig['p_value']:.4f} "
                         f"({'significant' if thresh_sig['significant'] else 'not significant'})")
        report.append("")
        
        # Character Position Analysis
        report.append("5. CHARACTER POSITION ANALYSIS")
        report.append("-" * 35)
        
        char_positions = ['char_0_correct', 'char_1_correct', 'char_2_correct', 
                         'char_3_correct', 'char_4_correct', 'char_5_correct', 'char_6_correct']
        
        for i, pos in enumerate(char_positions):
            if pos in self.char_df.columns:
                accuracy = self.char_df[pos].mean()
                report.append(f"  Position {i}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        report.append("")
        
        # False Positive Analysis
        report.append("6. ERROR ANALYSIS")
        report.append("-" * 18)
        
        fp_counts = self.fp_df['prediction_type'].value_counts()
        total = len(self.fp_df)
        
        for pred_type, count in fp_counts.items():
            percentage = count / total * 100
            report.append(f"  {pred_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Calculate character-level error rate
        total_chars = len(self.char_df) * 7
        correct_chars = self.char_df['characters_correct'].sum()
        char_error_rate = (total_chars - correct_chars) / total_chars
        
        report.append(f"\nCharacter-level error rate: {char_error_rate:.3f} ({char_error_rate*100:.1f}%)")
        report.append(f"Character-level accuracy: {1-char_error_rate:.3f} ({(1-char_error_rate)*100:.1f}%)")
        report.append("")
        
        # Statistical Significance
        report.append("7. STATISTICAL SIGNIFICANCE TESTS")
        report.append("-" * 38)
        
        if hasattr(self, 'significance_results'):
            for test_name, result in self.significance_results.items():
                report.append(f"\n{test_name.replace('_', ' ').title()}:")
                report.append(f"  Test: {result['test']}")
                report.append(f"  P-value: {result['p_value']:.4f}")
                report.append(f"  Result: {'Statistically significant' if result['significant'] else 'Not statistically significant'}")
                
                if 'no_threshold_mean' in result:
                    report.append(f"  No threshold mean: {result['no_threshold_mean']:.3f}")
                    report.append(f"  With threshold mean: {result['with_threshold_mean']:.3f}")
        report.append("")
        
        # Recommendations
        report.append("8. RECOMMENDATIONS FOR ACADEMIC PUBLICATION")
        report.append("-" * 48)
        
        best_method = best_methods.iloc[0]
        report.append(f"• Best preprocessing method: {best_method['preprocessing_method']}")
        report.append(f"• Optimal confidence threshold: {best_method['confidence_threshold']:.1f}")
        report.append(f"• Achieved accuracy: {best_method['full_plate_accuracy']:.3f}")
        
        if with_thresh_acc > no_thresh_acc:
            report.append("• Confidence thresholding improves performance")
        else:
            report.append("• Confidence thresholding does not improve performance")
        
        report.append(f"• Character-level accuracy is {overall_char_accuracy/overall_full_accuracy:.1f}x higher than full-plate accuracy")
        report.append("• Focus on character position-specific improvements")
        report.append("")
        
        # Publication Metrics
        report.append("9. KEY METRICS FOR PUBLICATION")
        report.append("-" * 35)
        
        report.append(f"• Dataset size: {self.valid_plates_df['filename'].nunique()} unique license plates")
        report.append(f"• Total experiments: {len(self.valid_plates_df):,} OCR tests")
        report.append(f"• Method combinations tested: {len(self.threshold_df)} (8 methods × 5 thresholds)")
        report.append(f"• Best accuracy achieved: {best_method['full_plate_accuracy']:.3f}")
        report.append(f"• Average character accuracy: {overall_char_accuracy:.3f}")
        report.append(f"• Statistical significance: {len([r for r in self.significance_results.values() if r['significant']])} of {len(self.significance_results)} tests significant")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.curated_dir, 'statistical_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print("ACADEMIC STATISTICAL ANALYSIS COMPLETE!")
        print("=" * 50)
        print(report_text)
        print(f"\nFull report saved to: {report_path}")
        
        return report_text
    
    def export_publication_tables(self):
        """Export publication-ready tables"""
        
        # Table 1: Method Performance Summary
        method_summary = self.threshold_df.groupby('preprocessing_method').agg({
            'full_plate_accuracy': ['mean', 'std', 'max'],
            'character_accuracy': 'mean',
            'detection_rate': 'mean',
            'total_tests': 'first'
        }).round(4)
        
        method_summary.columns = ['Accuracy_Mean', 'Accuracy_Std', 'Accuracy_Max', 
                                 'Char_Accuracy', 'Detection_Rate', 'Tests']
        
        # Table 2: Threshold Impact Analysis
        threshold_summary = self.threshold_df.groupby('confidence_threshold').agg({
            'full_plate_accuracy': ['mean', 'std'],
            'character_accuracy': 'mean',
            'detection_rate': 'mean'
        }).round(4)
        
        threshold_summary.columns = ['Accuracy_Mean', 'Accuracy_Std', 'Char_Accuracy', 'Detection_Rate']
        
        # Table 3: Top 10 Combinations
        top_combinations = self.threshold_df.nlargest(10, 'full_plate_accuracy')[
            ['preprocessing_method', 'confidence_threshold', 'full_plate_accuracy', 
             'character_accuracy', 'detection_rate']
        ].round(4)
        
        # Table 4: Character Position Analysis
        char_positions = ['char_0_correct', 'char_1_correct', 'char_2_correct', 
                         'char_3_correct', 'char_4_correct', 'char_5_correct', 'char_6_correct']
        
        position_analysis = pd.DataFrame({
            'Position': range(7),
            'Accuracy': [self.char_df[pos].mean() for pos in char_positions if pos in self.char_df.columns],
            'Character_Type': ['Letter', 'Letter', 'Letter', 'Number', 'Letter/Number', 'Number', 'Number']
        }).round(4)
        
        # Table 5: Error Analysis
        error_analysis = pd.DataFrame({
            'Error_Type': ['False_Positive', 'False_Negative', 'True_Positive'],
            'Count': [
                len(self.fp_df[self.fp_df['prediction_type'] == 'false_positive']),
                len(self.fp_df[self.fp_df['prediction_type'] == 'false_negative']),
                len(self.fp_df[self.fp_df['prediction_type'] == 'true_positive'])
            ],
            'Percentage': [
                len(self.fp_df[self.fp_df['prediction_type'] == 'false_positive']) / len(self.fp_df) * 100,
                len(self.fp_df[self.fp_df['prediction_type'] == 'false_negative']) / len(self.fp_df) * 100,
                len(self.fp_df[self.fp_df['prediction_type'] == 'true_positive']) / len(self.fp_df) * 100
            ]
        }).round(2)
        
        # Save all tables
        tables_dir = os.path.join(self.curated_dir, 'publication_tables')
        if not os.path.exists(tables_dir):
            os.makedirs(tables_dir)
        
        method_summary.to_csv(os.path.join(tables_dir, 'table1_method_performance.csv'))
        threshold_summary.to_csv(os.path.join(tables_dir, 'table2_threshold_impact.csv'))
        top_combinations.to_csv(os.path.join(tables_dir, 'table3_top_combinations.csv'), index=False)
        position_analysis.to_csv(os.path.join(tables_dir, 'table4_character_positions.csv'), index=False)
        error_analysis.to_csv(os.path.join(tables_dir, 'table5_error_analysis.csv'), index=False)
        
        # Create LaTeX tables for publication
        latex_tables = []
        
        # LaTeX Table 1: Method Performance
        latex_tables.append("% Table 1: Preprocessing Method Performance")
        latex_tables.append("\\begin{table}[htbp]")
        latex_tables.append("\\centering")
        latex_tables.append("\\caption{Performance comparison of preprocessing methods}")
        latex_tables.append("\\label{tab:method_performance}")
        latex_tables.append("\\begin{tabular}{lccccc}")
        latex_tables.append("\\hline")
        latex_tables.append("Method & Accuracy & Std Dev & Max Acc & Char Acc & Detection Rate \\\\")
        latex_tables.append("\\hline")
        
        for method, row in method_summary.iterrows():
            latex_tables.append(f"{method} & {row['Accuracy_Mean']:.3f} & {row['Accuracy_Std']:.3f} & {row['Accuracy_Max']:.3f} & {row['Char_Accuracy']:.3f} & {row['Detection_Rate']:.3f} \\\\")
        
        latex_tables.append("\\hline")
        latex_tables.append("\\end{tabular}")
        latex_tables.append("\\end{table}")
        latex_tables.append("")
        
        # Save LaTeX tables
        with open(os.path.join(tables_dir, 'latex_tables.tex'), 'w') as f:
            f.write('\n'.join(latex_tables))
        
        print(f"Publication tables exported to: {tables_dir}")
        
        return {
            'method_summary': method_summary,
            'threshold_summary': threshold_summary,
            'top_combinations': top_combinations,
            'position_analysis': position_analysis,
            'error_analysis': error_analysis
        }
    
    def run_complete_analysis(self):
        """Run the complete statistical analysis pipeline"""
        
        print("Starting comprehensive statistical analysis...")
        print("=" * 50)
        
        # Step 1: Filter to valid plates
        print("Step 1: Filtering to valid 7-character plates...")
        self.filter_valid_plates()
        
        # Step 2: Calculate character-level accuracy
        print("\nStep 2: Calculating character-level accuracy...")
        self.calculate_character_accuracy()
        
        # Step 3: Analyze threshold impact
        print("\nStep 3: Analyzing threshold impact...")
        self.analyze_threshold_impact()
        
        # Step 4: Calculate false positives
        print("\nStep 4: Calculating false positive rates...")
        self.calculate_false_positives()
        
        # Step 5: Statistical significance tests
        print("\nStep 5: Running statistical significance tests...")
        self.statistical_significance_tests()
        
        # Step 6: Create visualizations
        print("\nStep 6: Creating comprehensive visualizations...")
        self.create_individual_visualizations()
        
        # Step 7: Generate academic report
        print("\nStep 7: Generating academic report...")
        self.generate_academic_report()
        
        # Step 8: Export publication tables
        print("\nStep 8: Exporting publication-ready tables...")
        tables = self.export_publication_tables()
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("Generated files:")
        print(f"• Statistical analysis plot: {self.curated_dir}/statistical_analysis.png")
        print(f"• Academic report: {self.curated_dir}/statistical_analysis_report.txt")
        print(f"• Publication tables: {self.curated_dir}/publication_tables/")
        print(f"• LaTeX tables: {self.curated_dir}/publication_tables/latex_tables.tex")
        
        return {
            'valid_plates_df': self.valid_plates_df,
            'char_df': self.char_df,
            'threshold_df': self.threshold_df,
            'significance_results': self.significance_results,
            'publication_tables': tables
        }

# Usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = LicensePlateStatisticalAnalyzer('curated_dataset')
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\nKEY FINDINGS SUMMARY:")
    print("=" * 30)
    
    if 'threshold_comparison' in analyzer.significance_results:
        thresh_result = analyzer.significance_results['threshold_comparison']
        print(f"• Threshold vs No-threshold: {'Significant improvement' if thresh_result['significant'] else 'No significant difference'}")
        print(f"  - No threshold: {thresh_result['no_threshold_mean']:.3f}")
        print(f"  - With threshold: {thresh_result['with_threshold_mean']:.3f}")
    
    # Best method
    best_combo = analyzer.threshold_df.loc[analyzer.threshold_df['full_plate_accuracy'].idxmax()]
    print(f"• Best combination: {best_combo['preprocessing_method']} with threshold {best_combo['confidence_threshold']:.1f}")
    print(f"  - Full plate accuracy: {best_combo['full_plate_accuracy']:.3f}")
    
    # Character vs full accuracy
    overall_char = analyzer.char_df['character_accuracy'].mean()
    overall_full = analyzer.char_df['full_plate_correct'].mean()
    print(f"• Character accuracy: {overall_char:.3f} ({overall_char*100:.1f}%)")
    print(f"• Full plate accuracy: {overall_full:.3f} ({overall_full*100:.1f}%)")
    print(f"• Character accuracy is {overall_char/overall_full:.1f}x higher than full plate accuracy")
    
    print(f"\nReady for academic publication! 🎓📊")