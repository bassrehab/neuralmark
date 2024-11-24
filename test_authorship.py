import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from alphapunch.author import ImageAuthor
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import load_config, setup_logger, get_test_images, ImageManipulator


class RunManager:
    """Manage run-specific paths and identifiers."""

    def __init__(self, config: dict):
        self.config = config
        self.run_id = self._generate_run_id()
        self.run_paths = self._setup_run_paths()

    def _generate_run_id(self) -> str:
        """Generate unique run ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{unique_id}"

    def _setup_run_paths(self) -> dict:
        """Create and return run-specific paths."""
        paths = {}
        base_dirs = ['output', 'plots', 'reports']

        for dir_type in base_dirs:
            base_path = Path(self.config['directories'][dir_type])
            run_path = base_path / self.run_id
            run_path.mkdir(parents=True, exist_ok=True)
            paths[dir_type] = run_path

        return paths

    def get_path(self, dir_type: str, filename: str) -> Path:
        """Get full path for a file in a specific directory type."""
        return self.run_paths[dir_type] / filename


class AuthorshipTester:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the authorship tester with configuration."""
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)

        # Initialize run manager
        self.run_manager = RunManager(self.config)
        self.logger.info(f"Started new test run with ID: {self.run_manager.run_id}")

        self.author = ImageAuthor(
            private_key=self.config['private_key'],
            logger=self.logger,
            config=self.config
        )

        self.manipulator = ImageManipulator(self.config, self.logger)

    def run_authorship_tests(self) -> Dict[str, Any]:
        """Run complete authorship verification tests."""
        self.logger.info("Starting authorship verification tests...")

        test_cases = []
        start_time = time.time()

        try:
            # 1. Get test images
            original_images = get_test_images(
                self.config['testing']['num_original_images'],
                self.config,
                self.logger
            )

            # 2. Test original images
            self.logger.info("Testing with original images...")
            original_results = self._test_original_images(original_images)
            test_cases.extend(original_results)

            # 3. Test manipulated images
            self.logger.info("Testing with manipulated images...")
            manipulation_results = self._test_manipulated_images(original_images)
            test_cases.extend(manipulation_results)

            # 4. Test unrelated images
            self.logger.info("Testing with unrelated images...")
            unrelated_results = self._test_unrelated_images()
            test_cases.extend(unrelated_results)

            # Generate report
            report = self._generate_report(test_cases)

            # Add performance metrics
            report['performance'] = {
                'execution_time': time.time() - start_time,
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
            }

            # Save report and generate visualizations
            self._save_report(report)
            self._generate_visualizations(report)

            return report

        except Exception as e:
            self.logger.error(f"Error during authorship tests: {str(e)}")
            raise

    def run_cross_validation(self, k_folds: int = 5) -> Dict[str, Any]:
        """Run cross-validation testing."""
        self.logger.info(f"Starting {k_folds}-fold cross-validation...")

        try:
            # Get all test images
            all_images = get_test_images(
                self.config['testing']['num_original_images'],
                self.config,
                self.logger
            )

            # Initialize K-fold
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_results = []

            # Run each fold
            for fold, (train_idx, test_idx) in enumerate(kf.split(all_images), 1):
                self.logger.info(f"Processing fold {fold}/{k_folds}")

                # Split data
                train_images = [all_images[i] for i in train_idx]
                test_images = [all_images[i] for i in test_idx]

                # Train verifier (if applicable)
                authentic_pairs = self._generate_training_pairs(train_images)
                self.author.fingerprinter.train_verifier(authentic_pairs, [])

                # Test on fold
                fold_results = self._test_fold(test_images)
                cv_results.append(fold_results)

            # Aggregate results
            aggregated_results = self._aggregate_cv_results(cv_results)

            # Save CV report
            self._save_cv_report(aggregated_results)

            return aggregated_results

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def _test_original_images(self, original_images: List[str]) -> List[Dict]:
        """Test verification with original images."""
        results = []

        for img_path in tqdm(original_images, desc="Testing originals"):
            try:
                # Fingerprint the image
                fingerprinted_path = os.path.join(
                    self.config['directories']['output'],
                    f"fp_{Path(img_path).name}"
                )
                _, fingerprint = self.author.fingerprint_image(img_path, fingerprinted_path)

                # Verify the fingerprinted version
                is_owned, orig_path, similarity, mods = self.author.verify_ownership(fingerprinted_path)

                results.append({
                    'scenario': 'original',
                    'test_image': fingerprinted_path,
                    'source_image': img_path,
                    'verified': is_owned,
                    'similarity': similarity,
                    'modifications': mods,
                    'expected': True
                })

            except Exception as e:
                self.logger.error(f"Error testing original image {img_path}: {str(e)}")
                continue

        return results

    def _test_manipulated_images(self, original_images: List[str]) -> List[Dict]:
        """Test verification with manipulated versions of fingerprinted images."""
        results = []
        manipulations = [
            'blur', 'compress', 'rotate', 'crop', 'resize', 'noise'
        ]

        for img_path in tqdm(original_images, desc="Testing manipulations"):
            try:
                # First fingerprint the image
                fingerprinted_path = os.path.join(
                    self.config['directories']['output'],
                    f"fp_{Path(img_path).name}"
                )
                _, fingerprint = self.author.fingerprint_image(img_path, fingerprinted_path)

                # Load fingerprinted image
                fingerprinted_img = cv2.imread(fingerprinted_path)

                # Test each manipulation
                for manip_name in manipulations:
                    # Apply manipulation
                    manipulated_img = self.manipulator.apply_manipulation(
                        fingerprinted_img.copy(),
                        manip_name
                    )

                    # Save manipulated version
                    manip_path = os.path.join(
                        self.config['directories']['output'],
                        f"{manip_name}_{Path(img_path).name}"
                    )
                    cv2.imwrite(manip_path, manipulated_img)

                    # Verify ownership
                    is_owned, orig_path, similarity, mods = self.author.verify_ownership(manip_path)

                    results.append({
                        'scenario': f'manipulated_{manip_name}',
                        'test_image': manip_path,
                        'source_image': img_path,
                        'verified': is_owned,
                        'similarity': similarity,
                        'modifications': mods,
                        'expected': True
                    })

            except Exception as e:
                self.logger.error(f"Error testing manipulated image {img_path}: {str(e)}")
                continue

        return results

    def _test_unrelated_images(self) -> List[Dict]:
        """Test verification with unrelated images."""
        results = []

        try:
            unrelated_images = get_test_images(
                self.config['testing']['num_unrelated_images'],
                self.config,
                self.logger
            )

            for img_path in tqdm(unrelated_images, desc="Testing unrelated"):
                is_owned, orig_path, similarity, mods = self.author.verify_ownership(img_path)

                results.append({
                    'scenario': 'unrelated',
                    'test_image': img_path,
                    'source_image': None,
                    'verified': is_owned,
                    'similarity': similarity,
                    'modifications': mods,
                    'expected': False
                })

        except Exception as e:
            self.logger.error(f"Error testing unrelated images: {str(e)}")
            raise

        return results

    def _generate_report(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive report from test cases."""
        try:
            # Helper function to convert numpy types to Python native types
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            # Process test cases
            processed_cases = []
            for case in test_cases:
                processed_case = {
                    key: convert_to_serializable(value)
                    for key, value in case.items()
                }
                processed_cases.append(processed_case)

            # Generate report structure
            report = {
                'summary': {
                    'total_tests': len(processed_cases),
                    'successful_verifications': sum(1 for t in processed_cases if t['verified'] == t['expected']),
                    'false_positives': sum(1 for t in processed_cases if t['verified'] and not t['expected']),
                    'false_negatives': sum(1 for t in processed_cases if not t['verified'] and t['expected'])
                },
                'by_scenario': {},
                'test_cases': processed_cases,
                'config': self.config,
                'timestamp': datetime.now().isoformat(),
                'run_id': self.run_manager.run_id
            }

            # Group results by scenario
            for case in processed_cases:
                scenario = case['scenario']
                if scenario not in report['by_scenario']:
                    report['by_scenario'][scenario] = {
                        'total': 0,
                        'successful': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'average_similarity': 0.0,
                        'modifications': {},
                        'processing_times': []
                    }

                scenario_data = report['by_scenario'][scenario]
                scenario_data['total'] += 1

                if case['verified'] == case['expected']:
                    scenario_data['successful'] += 1
                elif case['verified'] and not case['expected']:
                    scenario_data['false_positives'] += 1
                elif not case['verified'] and case['expected']:
                    scenario_data['false_negatives'] += 1

                scenario_data['average_similarity'] += float(case['similarity'])

                # Track modifications
                for mod in case.get('modifications', []):
                    scenario_data['modifications'][mod] = \
                        scenario_data['modifications'].get(mod, 0) + 1

                if 'processing_time' in case:
                    scenario_data['processing_times'].append(float(case['processing_time']))

            # Calculate averages and rates
            for scenario, data in report['by_scenario'].items():
                if data['total'] > 0:
                    data['average_similarity'] /= data['total']
                    data['success_rate'] = (data['successful'] / data['total'] * 100)

                    # Calculate precision and recall
                    true_positives = data['successful']
                    false_positives = data['false_positives']
                    false_negatives = data['false_negatives']

                    data['precision'] = true_positives / (true_positives + false_positives) if (
                                                                                                       true_positives + false_positives) > 0 else 0
                    data['recall'] = true_positives / (true_positives + false_negatives) if (
                                                                                                    true_positives + false_negatives) > 0 else 0

                    # Calculate average processing time
                    if data['processing_times']:
                        data['average_processing_time'] = sum(data['processing_times']) / len(data['processing_times'])

                # Convert numpy types to native Python types
                data['average_similarity'] = float(data['average_similarity'])
                data['success_rate'] = float(data['success_rate'])
                if 'average_processing_time' in data:
                    data['average_processing_time'] = float(data['average_processing_time'])

            return report

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise

    def _calculate_performance_metrics(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        metrics = {
            'execution_times': {
                'mean': np.mean([case.get('execution_time', 0) for case in test_cases]),
                'std': np.std([case.get('execution_time', 0) for case in test_cases]),
                'max': np.max([case.get('execution_time', 0) for case in test_cases]),
                'min': np.min([case.get('execution_time', 0) for case in test_cases])
            },
            'memory_usage': {
                'peak': psutil.Process().memory_info().rss / 1024 / 1024  # MB
            },
            'accuracy_metrics': {
                'precision': self._calculate_precision(test_cases),
                'recall': self._calculate_recall(test_cases),
                'f1_score': self._calculate_f1_score(test_cases)
            }
        }
        return metrics

    def _calculate_precision(self, test_cases: List[Dict]) -> float:
        """Calculate precision metric."""
        true_positives = sum(1 for t in test_cases if t['verified'] and t['expected'])
        false_positives = sum(1 for t in test_cases if t['verified'] and not t['expected'])
        if true_positives + false_positives == 0:
            return 0.0
        return true_positives / (true_positives + false_positives)

    def _calculate_recall(self, test_cases: List[Dict]) -> float:
        """Calculate recall metric."""
        true_positives = sum(1 for t in test_cases if t['verified'] and t['expected'])
        false_negatives = sum(1 for t in test_cases if not t['verified'] and t['expected'])
        if true_positives + false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + false_negatives)

    def _calculate_f1_score(self, test_cases: List[Dict]) -> float:
        """Calculate F1 score."""
        precision = self._calculate_precision(test_cases)
        recall = self._calculate_recall(test_cases)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _generate_visualizations(self, report: Dict[str, Any]) -> None:
        """Generate visualization plots organized by run ID."""
        try:
            # Use seaborn style with matplotlib
            import seaborn as sns
            sns.set_theme(style="whitegrid")

            # Create plot directories
            plots_base_dir = self.run_manager.run_paths['plots']
            plot_categories = ['performance', 'accuracy', 'modifications']
            plot_dirs = {
                category: plots_base_dir / category
                for category in plot_categories
            }

            # Create directories
            for dir_path in plot_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)

            # 1. Performance Plots
            self._generate_performance_plots(report, plot_dirs['performance'])

            # 2. Accuracy Plots
            self._generate_accuracy_plots(report, plot_dirs['accuracy'])

            # 3. Modification Plots
            self._generate_modification_plots(report, plot_dirs['modifications'])

            self.logger.info(f"Generated visualizations in {plots_base_dir}")

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def _generate_performance_plots(self, report: Dict[str, Any], plot_dir: Path) -> None:
        """Generate performance-related plots."""
        try:
            # Processing Time Plot
            plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
            scenarios = list(report['by_scenario'].keys())
            proc_times = [data.get('average_processing_time', 0)
                          for data in report['by_scenario'].values()]

            sns.barplot(x=scenarios, y=proc_times)
            plt.title('Processing Time by Scenario')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Time (seconds)')
            plt.tight_layout()
            plt.savefig(plot_dir / 'processing_times.png',
                        dpi=self.config['visualization']['dpi'])
            plt.close()

            # Memory Usage Plot (if available)
            if 'memory_usage' in report.get('performance_metrics', {}):
                plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
                memory_data = report['performance_metrics']['memory_usage']
                sns.lineplot(data=memory_data)
                plt.title('Memory Usage Over Time')
                plt.xlabel('Test Stage')
                plt.ylabel('Memory Usage (MB)')
                plt.tight_layout()
                plt.savefig(plot_dir / 'memory_usage.png',
                            dpi=self.config['visualization']['dpi'])
                plt.close()

        except Exception as e:
            self.logger.error(f"Error generating performance plots: {str(e)}")

    def _generate_accuracy_plots(self, report: Dict[str, Any], plot_dir: Path) -> None:
        """Generate accuracy-related plots."""
        try:
            # Success Rates Plot
            plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
            scenarios = list(report['by_scenario'].keys())
            success_rates = [report['by_scenario'][s]['success_rate']
                             for s in scenarios]

            sns.barplot(x=scenarios, y=success_rates)
            plt.title('Success Rates by Scenario')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Success Rate (%)')
            plt.tight_layout()
            plt.savefig(plot_dir / 'success_rates.png',
                        dpi=self.config['visualization']['dpi'])
            plt.close()

            # Similarity Distribution
            plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
            similarities = [case['similarity'] for case in report['test_cases']]
            sns.histplot(data=similarities, bins=30)
            plt.title('Distribution of Similarity Scores')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(plot_dir / 'similarity_distribution.png',
                        dpi=self.config['visualization']['dpi'])
            plt.close()

            # Precision-Recall Plot
            plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
            precisions = []
            recalls = []
            labels = []
            for scenario, data in report['by_scenario'].items():
                if 'precision' in data and 'recall' in data:
                    precisions.append(data['precision'])
                    recalls.append(data['recall'])
                    labels.append(scenario)

            plt.scatter(recalls, precisions)
            for i, label in enumerate(labels):
                plt.annotate(label, (recalls[i], precisions[i]))
            plt.title('Precision vs Recall by Scenario')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_dir / 'precision_recall.png',
                        dpi=self.config['visualization']['dpi'])
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating accuracy plots: {str(e)}")

    def _generate_modification_plots(self, report: Dict[str, Any], plot_dir: Path) -> None:
        """Generate modification-related plots."""
        try:
            # Modification Types Plot
            plt.figure(figsize=tuple(self.config['visualization']['figure_size']))
            all_mods = {}
            for scenario in report['by_scenario'].values():
                for mod, count in scenario.get('modifications', {}).items():
                    all_mods[mod] = all_mods.get(mod, 0) + count

            if all_mods:
                mods = list(all_mods.keys())
                counts = list(all_mods.values())
                sns.barplot(x=mods, y=counts)
                plt.title('Types of Detected Modifications')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(plot_dir / 'modification_types.png',
                            dpi=self.config['visualization']['dpi'])
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating modification plots: {str(e)}")

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save report to file with proper type conversion."""
        try:
            # Create report directory if it doesn't exist
            report_dir = self.run_manager.run_paths['reports']
            report_dir.mkdir(parents=True, exist_ok=True)

            # Save detailed JSON report
            report_path = report_dir / 'test_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Save summary report
            summary_path = report_dir / 'summary.txt'
            with open(summary_path, 'w') as f:
                f.write(f"Test Run Summary\n")
                f.write(f"===============\n")
                f.write(f"Run ID: {report['run_id']}\n")
                f.write(f"Timestamp: {report['timestamp']}\n\n")

                # Overall statistics
                f.write("Overall Statistics:\n")
                f.write(f"Total Tests: {report['summary']['total_tests']}\n")
                f.write(f"Successful Verifications: {report['summary']['successful_verifications']}\n")
                f.write(f"False Positives: {report['summary']['false_positives']}\n")
                f.write(f"False Negatives: {report['summary']['false_negatives']}\n\n")

                # Results by scenario
                f.write("Results by Scenario:\n")
                for scenario, data in report['by_scenario'].items():
                    f.write(f"\n{scenario.upper()}:\n")
                    f.write(f"Success Rate: {data['success_rate']:.2f}%\n")
                    f.write(f"Average Similarity: {data['average_similarity']:.2%}\n")
                    if data.get('modifications'):
                        f.write("Detected Modifications:\n")
                        for mod, count in data['modifications'].items():
                            f.write(f"  - {mod}: {count}\n")

            self.logger.info(f"Saved test report to {report_path}")
            self.logger.info(f"Saved summary to {summary_path}")

        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            raise


if __name__ == "__main__":
    # Create and run tester
    tester = AuthorshipTester()

    try:
        # Run main tests
        results = tester.run_authorship_tests()

        # Run cross-validation if enabled in config
        if tester.config.get('testing', {}).get('cross_validation', {}).get('enabled', False):
            cv_results = tester.run_cross_validation()

    except Exception as e:
        logging.error(f"Error running tests: {str(e)}")
        raise
