import json
import multiprocessing
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2
from tqdm import tqdm

from neuralmark.algorithm import create_neural_mark
from neuralmark.author import ImageAuthor
from neuralmark.utils import (
    load_config,
    setup_logger,
    get_test_images,
    ImageManipulator,
    visualize_attention_maps,
    plot_test_results,
    create_comparison_grid
)


class RunManager:
    def __init__(self, config: dict):
        self.config = config
        self.run_id = self._generate_run_id()
        self.base_output = Path(config['directories']['base_output'])
        self.run_dir = self.base_output / self.run_id
        self.run_paths = self._setup_run_paths()

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{unique_id}"

    def _setup_run_paths(self) -> dict:
        """Setup run-specific directory structure."""
        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        # Run-specific directories
        subdirs = ['fingerprinted', 'manipulated', 'plots', 'reports', 'test']

        for dir_name in subdirs:
            dir_path = self.run_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            paths[dir_name] = dir_path

            # Create algorithm-specific subdirectories if comparison mode is enabled
            if self.config['algorithm_selection'].get('enable_comparison', False):
                (dir_path / 'amdf').mkdir(exist_ok=True)
                (dir_path / 'cdha').mkdir(exist_ok=True)

        return paths


class AuthorshipTester:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)

        # Initialize test management
        self.run_manager = RunManager(self.config)
        self.logger.info(f"{__name__}.__init__ - Started new test run with ID: {self.run_manager.run_id}")

        # Update config with run ID
        self.config['run_id'] = self.run_manager.run_id

        # Initialize test paths
        self._setup_paths()

        # Initialize system components
        self._init_authors()
        self.manipulator = ImageManipulator(self.config, self.logger)
        self.max_workers = 1 if not self.config['resources'].get('parallel_processing', False) else self.config[
            'resources'].get('num_workers', multiprocessing.cpu_count())

    def _setup_paths(self):
        """Setup all necessary paths for the test run."""
        # Update config with run-specific paths
        self.config['directories'].update({
            'fingerprinted': str(self.run_manager.run_paths['fingerprinted']),
            'manipulated': str(self.run_manager.run_paths['manipulated']),
            'plots': str(self.run_manager.run_paths['plots']),
            'reports': str(self.run_manager.run_paths['reports']),
            'test': str(self.run_manager.run_paths['test'])
        })

        # Log directory setup
        self.logger.info(f"Test run directory: {self.run_manager.run_dir}")
        for name, path in self.run_manager.run_paths.items():
            self.logger.debug(f"Created {name} directory: {path}")

    def _init_authors(self):
        """Initialize authors for selected algorithms."""
        self.authors = {}

        # Create primary author
        primary_algo = self.config['algorithm_selection']['type']
        self.authors[primary_algo] = ImageAuthor(
            private_key=self.config['private_key'],
            logger=self.logger,
            config=self.config
        )

        # Create comparison author if enabled
        if self.config['algorithm_selection'].get('enable_comparison', False):
            comparison_algo = 'amdf' if primary_algo == 'cdha' else 'cdha'
            comparison_config = self.config.copy()
            comparison_config['algorithm_selection']['type'] = comparison_algo
            self.authors[comparison_algo] = ImageAuthor(
                private_key=self.config['private_key'],
                logger=self.logger,
                config=comparison_config
            )

    def _save_test_metadata(self):
        """Save test run metadata."""
        metadata = {
            'run_id': self.run_manager.run_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'algorithm': self.config['algorithm_selection']['type'],
            'comparison_enabled': self.config['algorithm_selection'].get('enable_comparison', False)
        }

        metadata_path = self.run_manager.run_paths['reports'] / 'run_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_image_batch(self, image_paths: List[str], process_func) -> List[Dict]:
        """Process images in parallel."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(process_func, path): path
                              for path in image_paths}

            for future in tqdm(as_completed(future_to_path),
                               total=len(image_paths),
                               desc="Processing images"):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.extend(result if isinstance(result, list) else [result])
                except Exception as e:
                    self.logger.error(f"Error processing {path}: {str(e)}")

        return results

    def _generate_comparison_report(self, algorithm_reports: Dict) -> Dict:
        """Generate comparison report between algorithms."""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_manager.run_id,
            'comparisons': {}
        }

        metrics = ['success_rate', 'false_positives', 'false_negatives', 'execution_time']

        for metric in metrics:
            comparison['comparisons'][metric] = {
                algo: report['summary'].get(metric, 0)
                for algo, report in algorithm_reports.items()
            }

        # Calculate relative performance
        for metric in metrics:
            values = [v for v in comparison['comparisons'][metric].values()]
            if values:
                best = max(values)
                for algo in comparison['comparisons'][metric]:
                    current = comparison['comparisons'][metric][algo]
                    comparison['comparisons'][metric][f'{algo}_relative'] = current / best

        return comparison

    def _save_comparison_report(self, report: Dict):
        """Save comparison report to file."""
        report_dir = self.run_manager.run_paths['reports']

        # Save JSON report
        report_path = report_dir / 'algorithm_comparison.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate comparison visualizations
        self._generate_comparison_visualizations(report)

    def _generate_comparison_visualizations(self, report: Dict):
        """Generate visualizations comparing algorithm performance."""
        plots_dir = self.run_manager.run_paths['plots']

        metrics = ['success_rate', 'false_positives', 'false_negatives', 'execution_time']

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            algorithms = list(report['comparisons'][metric].keys())
            values = [report['comparisons'][metric][algo] for algo in algorithms]

            plt.bar(algorithms, values)
            plt.title(f'Algorithm Comparison: {metric.replace("_", " ").title()}')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(plots_dir / f'comparison_{metric}.png')
            plt.close()

    def run_authorship_tests(self) -> Dict[str, Any]:
        """Run complete test suite with parallel processing."""
        test_cases = []
        start_time = time.time()

        try:
            # Get and split dataset
            total_images = self.config['testing']['total_images']
            train_ratio = self.config['testing']['train_ratio']
            train_size = int(total_images * train_ratio)

            all_images = get_test_images(total_images, self.config, self.logger)
            train_images = all_images[:train_size]
            test_images = all_images[train_size:]

            # Process training images
            self.logger.info("Processing training images...")
            test_cases.extend(self.process_image_batch(
                train_images,
                self._process_training_image
            ))

            # Process test images
            self.logger.info("Processing test images...")
            test_cases.extend(self.process_image_batch(
                test_images,
                self._process_test_image
            ))

            # Generate reports for each algorithm
            reports = {}
            for algo in self.authors.keys():
                algo_cases = [case for case in test_cases if case.get('algorithm') == algo]
                if algo_cases:
                    reports[algo] = self._generate_report(algo_cases)
                    reports[algo]['performance'] = {
                        'execution_time': time.time() - start_time,
                        'num_workers': self.max_workers
                    }
                    self._save_report(reports[algo], algo)

            # Generate comparison report if needed
            if len(self.authors) > 1:
                comparison_report = self._generate_comparison_report(reports)
                self._save_comparison_report(comparison_report)

            # Save test metadata
            self._save_test_metadata()

            return reports

        except Exception as e:
            self.logger.error(f"Error during authorship tests: {str(e)}")
            raise

    def run_cross_validation(self) -> Dict[str, Any]:
        """Run k-fold cross-validation for each algorithm."""
        results = {}

        for algo, author in self.authors.items():
            self.logger.info(f"Running cross-validation for {algo}")
            results[algo] = self._run_single_algorithm_cross_validation(algo)

        # Generate comparison if multiple algorithms
        if len(self.authors) > 1:
            results['comparison'] = self._compare_cross_validation_results(results)

        return results

    def _run_single_algorithm_cross_validation(self, algorithm: str) -> Dict[str, Any]:
        """Run cross-validation for a single algorithm."""
        k_folds = self.config['testing'].get('k_folds', 5)
        total_images = self.config['testing']['total_images']

        all_images = get_test_images(total_images, self.config, self.logger)
        fold_size = len(all_images) // k_folds
        fold_results = []

        for fold in range(k_folds):
            test_indices = slice(fold * fold_size, (fold + 1) * fold_size)
            test_images = all_images[test_indices]
            train_images = all_images[:fold * fold_size] + all_images[(fold + 1) * fold_size:]

            # Process fold
            train_results = self.process_image_batch(
                train_images,
                lambda x: self._process_training_image(x, algorithm)
            )
            test_results = self.process_image_batch(
                test_images,
                lambda x: self._process_test_image(x, algorithm)
            )

            # Generate fold report
            fold_report = self._generate_report(train_results + test_results, algorithm)
            fold_results.append(fold_report['summary']['success_rate'])

        # Calculate statistics
        cv_results = {
            'fold_success_rates': fold_results,
            'average_success_rate': float(np.mean(fold_results)),
            'std_success_rate': float(np.std(fold_results)),
            'min_success_rate': float(np.min(fold_results)),
            'max_success_rate': float(np.max(fold_results)),
            'k_folds': k_folds,
            'algorithm': algorithm,
            'timestamp': datetime.now().isoformat()
        }

        return cv_results

    def _process_training_image(self, img_path: str) -> List[Dict]:
        """Process a single training image."""
        results = []
        for algo, author in self.authors.items():
            try:
                # Define paths
                fp_path = str(self.run_manager.run_paths['fingerprinted'] / algo / Path(img_path).name)

                # Generate fingerprint
                fingerprinted_img, fingerprint = author.fingerprint_image(img_path, fp_path)

                # Test original
                results.append(self._test_original_image(fp_path, img_path, algo))

                # Test manipulated versions
                results.extend(self._test_manipulated_images(fp_path, img_path, algo))

            except Exception as e:
                self.logger.error(f"Error processing {algo} training image {img_path}: {str(e)}")

        return results

    def _process_test_image(self, img_path: str) -> Dict:
        """Process a single test image."""
        results = []

        for algo, author in self.authors.items():
            try:
                # Verify ownership (should return False)
                is_owned, orig_path, similarity, mods = author.verify_ownership(img_path)

                results.append({
                    'scenario': 'unrelated',
                    'test_image': img_path,
                    'source_image': None,
                    'verified': is_owned,
                    'similarity': similarity,
                    'modifications': mods,
                    'expected': False,
                    'algorithm': algo
                })

            except Exception as e:
                self.logger.error(f"Error testing unrelated image with {algo}: {str(e)}")
                results.append({
                    'scenario': 'unrelated',
                    'test_image': img_path,
                    'source_image': None,
                    'verified': False,
                    'similarity': 0.0,
                    'modifications': [],
                    'expected': False,
                    'algorithm': algo,
                    'error': str(e)
                })

        return results

    def _test_original_image(self, fp_path: str, orig_path: str, algorithm: str) -> Dict:
        """Test an original fingerprinted image."""
        try:
            # Verify ownership using appropriate author
            author = self.authors[algorithm]
            is_owned, orig_path_verified, similarity, mods = author.verify_ownership(fp_path)

            return {
                'scenario': 'original',
                'test_image': fp_path,
                'source_image': orig_path,
                'verified': is_owned,
                'similarity': similarity,
                'modifications': mods,
                'expected': True,
                'algorithm': algorithm
            }
        except Exception as e:
            self.logger.error(f"Error testing original image {fp_path}: {str(e)}")
            raise

    def _test_manipulated_images(self, fp_path: str, orig_path: str, algorithm: str) -> List[Dict]:
        """Test manipulated versions of a fingerprinted image."""
        results = []
        try:
            # Load fingerprinted image
            fp_img = cv2.imread(fp_path)
            if fp_img is None:
                raise ValueError(f"Could not read fingerprinted image: {fp_path}")

            author = self.authors[algorithm]

            # Test each manipulation type
            for manip_name in self.config['testing']['manipulations']:
                manipulated_img = self.manipulator.apply_manipulation(fp_img, manip_name)

                # Save manipulated image in algorithm-specific directory
                manip_path = str(self.run_manager.run_paths['manipulated'] / algorithm /
                                 f"{manip_name}_{Path(fp_path).name}")
                cv2.imwrite(manip_path, manipulated_img)

                # Verify ownership of manipulated version
                is_owned, orig_path_found, similarity, mods = author.verify_ownership(manip_path)

                results.append({
                    'scenario': f'manipulated_{manip_name}',
                    'test_image': manip_path,
                    'source_image': orig_path,
                    'verified': is_owned,
                    'similarity': similarity,
                    'modifications': mods,
                    'expected': True,
                    'algorithm': algorithm
                })

        except Exception as e:
            self.logger.error(f"Error in manipulation test: {str(e)}")
            raise

        return results

    def _test_unrelated_image(self, img_path: str) -> Dict:
        """Test a single unrelated image with all algorithms."""
        results = []

        for algo, author in self.authors.items():
            try:
                # Verify ownership (should return False)
                is_owned, orig_path, similarity, mods = author.verify_ownership(img_path)

                results.append({
                    'scenario': 'unrelated',
                    'test_image': img_path,
                    'source_image': None,
                    'verified': is_owned,
                    'similarity': similarity,
                    'modifications': mods,
                    'expected': False,
                    'algorithm': algo
                })

            except Exception as e:
                self.logger.error(f"Error testing unrelated image with {algo}: {str(e)}")
                results.append({
                    'scenario': 'unrelated',
                    'test_image': img_path,
                    'source_image': None,
                    'verified': False,
                    'similarity': 0.0,
                    'modifications': [],
                    'expected': False,
                    'algorithm': algo,
                    'error': str(e)
                })

        return results

    def _test_unrelated_images(self, test_images: List[str]) -> List[Dict]:
        """
        Test unrelated images (should not be identified as fingerprinted).

        Args:
            test_images: List of paths to test images

        Returns:
            List[Dict]: List of test results for each image
        """
        results = []

        for img_path in tqdm(test_images, desc="Testing unrelated"):
            if not Path(img_path).exists():
                self.logger.error(f"Image path does not exist: {img_path}")
                results.append({
                    'scenario': 'unrelated',
                    'test_image': img_path,
                    'source_image': None,
                    'verified': False,
                    'similarity': 0.0,
                    'modifications': [],
                    'expected': False,
                    'error': 'Image file not found'
                })
                continue

            try:
                result = self._test_unrelated_image(img_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error testing unrelated image {img_path}: {str(e)}")
                results.append({
                    'scenario': 'unrelated',
                    'test_image': img_path,
                    'source_image': None,
                    'verified': False,
                    'similarity': 0.0,
                    'modifications': [],
                    'expected': False,
                    'error': str(e)
                })

        return results

    def _generate_report(self, test_cases: List[Dict], algorithm: str = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if algorithm:
            test_cases = [case for case in test_cases if case.get('algorithm') == algorithm]

        report = {
            'summary': {
                'total_tests': len(test_cases),
                'successful_verifications': sum(1 for t in test_cases if t['verified'] == t['expected']),
                'false_positives': sum(1 for t in test_cases if t['verified'] and not t['expected']),
                'false_negatives': sum(1 for t in test_cases if not t['verified'] and t['expected'])
            },
            'by_scenario': {},
            'test_cases': test_cases,
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_manager.run_id,
            'algorithm': algorithm
        }

        # Calculate success rate
        if report['summary']['total_tests'] > 0:
            report['summary']['success_rate'] = (
                    report['summary']['successful_verifications'] /
                    report['summary']['total_tests'] * 100
            )

        # Group results by scenario
        for case in test_cases:
            scenario = case['scenario']
            if scenario not in report['by_scenario']:
                report['by_scenario'][scenario] = {
                    'total': 0,
                    'successful': 0,
                    'success_rate': 0,
                    'average_similarity': 0,
                    'modifications': {}
                }

            scenario_data = report['by_scenario'][scenario]
            scenario_data['total'] += 1
            if case['verified'] == case['expected']:
                scenario_data['successful'] += 1
            scenario_data['average_similarity'] += case['similarity']

            for mod in case.get('modifications', []):
                scenario_data['modifications'][mod] = scenario_data['modifications'].get(mod, 0) + 1

        # Calculate averages for each scenario
        for data in report['by_scenario'].values():
            if data['total'] > 0:
                data['success_rate'] = (data['successful'] / data['total']) * 100
                data['average_similarity'] = data['average_similarity'] / data['total']

        return report

    def _save_algorithm_specific_report(self, report: Dict[str, Any], algorithm: str):
        """Save algorithm-specific report and visualizations."""
        report_dir = self.run_manager.run_paths['reports'] / algorithm
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_path = report_dir / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save summary
        summary_path = report_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            self._write_summary(f, report)

        # Generate visualizations
        plots_dir = self.run_manager.run_paths['plots'] / algorithm
        plots_dir.mkdir(parents=True, exist_ok=True)
        self._generate_visualizations(report, plots_dir)

        self.logger.info(f"Saved {algorithm} test report to {report_dir}")

    def _save_report(self, report: Dict[str, Any], algorithm: str):
        """Save test report to files."""
        report_dir = self.run_manager.run_paths['reports']
        if algorithm:
            report_dir = report_dir / algorithm
            report_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_path = report_dir / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate visualizations
        plot_test_results(report, self.run_manager.run_dir, self.logger)

        self.logger.info(f"Saved test report to {report_path}")

    def _write_summary(self, file, report: Dict[str, Any]) -> None:
        """Write formatted summary to file."""
        file.write(f"Test Run Summary\n")
        file.write(f"===============\n")
        file.write(f"Run ID: {report['run_id']}\n")
        file.write(f"Timestamp: {report['timestamp']}\n\n")

        file.write("Overall Statistics:\n")
        for key, value in report['summary'].items():
            file.write(f"{key.replace('_', ' ').title()}: {value}\n")

        file.write("\nResults by Scenario:\n")
        for scenario, data in report['by_scenario'].items():
            file.write(f"\n{scenario.upper()}:\n")
            file.write(f"Success Rate: {data['success_rate']:.2f}%\n")
            file.write(f"Average Similarity: {data['average_similarity']:.2%}\n")
            if data['modifications']:
                file.write("Detected Modifications:\n")
                for mod, count in data['modifications'].items():
                    file.write(f"  - {mod}: {count}\n")

    def _generate_visualizations(self, report: Dict[str, Any]) -> None:
        """Generate visualization plots for test results."""
        plots_dir = self.run_manager.run_paths['plots']

        # Set style with default matplotlib instead of seaborn
        plt.style.use('default')
        # Set custom style parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        # 1. Success rates by scenario
        plt.figure(figsize=(12, 6))
        scenarios = list(report['by_scenario'].keys())
        success_rates = [data['success_rate'] for data in report['by_scenario'].values()]

        plt.bar(scenarios, success_rates, color='royalblue', alpha=0.7)
        plt.title('Success Rates by Scenario', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Success Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'success_rates.png')
        plt.close()

        # 2. Similarity distributions
        plt.figure(figsize=(12, 6))
        similarities = [case['similarity'] for case in report['test_cases']]
        plt.hist(similarities, bins=30, color='royalblue', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Similarity Scores', pad=20)
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'similarity_distribution.png')
        plt.close()

        # 3. Modification types
        all_mods = {}
        for case in report['test_cases']:
            for mod in case.get('modifications', []):
                all_mods[mod] = all_mods.get(mod, 0) + 1

        if all_mods:
            plt.figure(figsize=(12, 6))
            mods = list(all_mods.keys())
            counts = list(all_mods.values())

            plt.bar(mods, counts, color='royalblue', alpha=0.7)
            plt.title('Types of Detected Modifications', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'modification_types.png')
            plt.close()

        # 4. Error timeline
        plt.figure(figsize=(12, 6))
        test_indices = range(len(report['test_cases']))
        errors = [1 if case['verified'] != case['expected'] else 0
                  for case in report['test_cases']]
        plt.plot(test_indices, errors, 'r.', markersize=10, alpha=0.7)
        plt.title('Errors Over Time', pad=20)
        plt.xlabel('Test Case Index')
        plt.ylabel('Error (0=correct, 1=error)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'error_timeline.png')
        plt.close()

        self.logger.info(f"Generated visualizations in {plots_dir}")

    def _plot_success_rates(self, report: Dict, plots_dir: Path) -> None:
        plt.figure(figsize=(12, 6))
        scenarios = list(report['by_scenario'].keys())
        success_rates = [data['success_rate'] for data in report['by_scenario'].values()]

        sns.barplot(x=scenarios, y=success_rates)
        plt.title('Success Rates by Scenario')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Success Rate (%)')
        plt.tight_layout()
        plt.savefig(plots_dir / 'success_rates.png')
        plt.close()

    def _plot_similarity_distributions(self, report: Dict, plots_dir: Path) -> None:
        plt.figure(figsize=(12, 6))
        similarities = [case['similarity'] for case in report['test_cases']]
        sns.histplot(data=similarities, bins=30)
        plt.title('Distribution of Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(plots_dir / 'similarity_distribution.png')
        plt.close()

    def _plot_modification_types(self, report: Dict, plots_dir: Path) -> None:
        # Collect all modifications
        all_mods = {}
        for scenario_data in report['by_scenario'].values():
            for mod, count in scenario_data.get('modifications', {}).items():
                all_mods[mod] = all_mods.get(mod, 0) + count

        if all_mods:
            plt.figure(figsize=(12, 6))
            mods = list(all_mods.keys())
            counts = list(all_mods.values())

            sns.barplot(x=mods, y=counts)
            plt.title('Types of Detected Modifications')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(plots_dir / 'modification_types.png')
            plt.close()


if __name__ == "__main__":
    tester = AuthorshipTester()
    tester.run_authorship_tests()
