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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from alphapunch.author import ImageAuthor
from alphapunch.utils import load_config, setup_logger, get_test_images, ImageManipulator


class RunManager:
    def __init__(self, config: dict):
        self.config = config
        self.run_id = self._generate_run_id()
        self.run_paths = self._setup_run_paths()

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{unique_id}"

    def _setup_run_paths(self) -> dict:
        paths = {}
        for dir_type in ['output', 'plots', 'reports', 'fingerprinted', 'manipulated', 'test']:
            base_path = Path(self.config['directories'].get(dir_type, dir_type))
            run_path = base_path / self.run_id
            run_path.mkdir(parents=True, exist_ok=True)
            paths[dir_type] = run_path
        return paths


class AuthorshipTester:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)

        # Initialize test management
        self.run_manager = RunManager(self.config)
        self.logger.info(f"{__name__}.__init__ - Started new test run with ID: {self.run_manager.run_id}")

        # Initialize system components
        self.author = ImageAuthor(
            private_key=self.config['private_key'],
            logger=self.logger,
            config=self.config
        )

        self.manipulator = ImageManipulator(self.config, self.logger)

        self.max_workers = self.config.get('resources', {}).get('num_workers',
                                                                multiprocessing.cpu_count())

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

            # Process training images in parallel
            self.logger.info("Processing training images...")
            test_cases.extend(self.process_image_batch(
                train_images,
                self._process_training_image
            ))

            # Process test images in parallel
            self.logger.info("Processing test images...")
            test_cases.extend(self.process_image_batch(
                test_images,
                self._process_test_image
            ))

            # Generate report
            report = self._generate_report(test_cases)
            report['performance'] = {
                'execution_time': time.time() - start_time,
                'num_workers': self.max_workers
            }

            self._save_report(report)
            self._generate_visualizations(report)

            return report

        except Exception as e:
            self.logger.error(f"Error during authorship tests: {str(e)}")
            raise

    def run_cross_validation(self) -> Dict[str, Any]:
        """
        Run k-fold cross-validation on the test suite.

        Returns:
            Dict[str, Any]: Cross-validation results including success rates and statistics
        """
        try:
            # Get dataset
            total_images = self.config['testing']['total_images']
            k_folds = self.config['testing'].get('k_folds', 5)

            # Get all images
            all_images = get_test_images(total_images, self.config, self.logger)

            # Initialize results storage
            fold_results = []
            fold_size = len(all_images) // k_folds

            self.logger.info(f"Starting {k_folds}-fold cross-validation")

            # Run k-fold cross validation
            for fold in range(k_folds):
                self.logger.info(f"Processing fold {fold + 1}/{k_folds}")

                # Split data for this fold
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size

                # Use current fold as test set, rest as training set
                test_images = all_images[start_idx:end_idx]
                train_images = all_images[:start_idx] + all_images[end_idx:]

                # Process training images
                train_results = self.process_image_batch(
                    train_images,
                    self._process_training_image
                )

                # Process test images
                test_results = self.process_image_batch(
                    test_images,
                    self._process_test_image
                )

                # Generate report for this fold
                fold_report = self._generate_report(train_results + test_results)
                fold_results.append(fold_report['summary']['successful_verifications'] /
                                    fold_report['summary']['total_tests'] * 100)

            # Calculate cross-validation statistics
            success_rates = np.array(fold_results)
            cv_results = {
                'fold_success_rates': fold_results,
                'average_success_rate': float(np.mean(success_rates)),
                'std_success_rate': float(np.std(success_rates)),
                'min_success_rate': float(np.min(success_rates)),
                'max_success_rate': float(np.max(success_rates)),
                'k_folds': k_folds,
                'timestamp': datetime.now().isoformat(),
                'run_id': self.run_manager.run_id
            }

            # Save cross-validation results
            cv_path = self.run_manager.run_paths['reports'] / 'cross_validation_results.json'
            with open(cv_path, 'w') as f:
                json.dump(cv_results, f, indent=2)

            self.logger.info(f"Cross-validation complete. Results saved to {cv_path}")
            return cv_results

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def _process_training_image(self, img_path: str) -> List[Dict]:
        """
        Process a single training image through fingerprinting and testing.

        Args:
            img_path: Path to the source image

        Returns:
            List[Dict]: List of test results for original and manipulated versions
        """
        if not Path(img_path).exists():
            self.logger.error(f"Image path does not exist: {img_path}")
            return []

        # Define fp_path outside try block
        fp_path = str(self.run_manager.run_paths['fingerprinted'] / Path(img_path).name)

        try:
            # Create fingerprinted version
            fingerprinted_img, fingerprint = self.author.fingerprint_image(img_path, fp_path)

            if fingerprinted_img is None:
                self.logger.error(f"Failed to fingerprint image: {img_path}")
                return []

            # Process original and manipulated versions
            results = []

            # Test original
            original_result = self._test_original_image(fp_path, img_path)
            if original_result:
                results.append(original_result)

            # Test manipulated versions
            manipulated_results = self._test_manipulated_images(fp_path, img_path)
            results.extend(manipulated_results)

            return results

        except Exception as e:
            self.logger.error(f"Error processing training image {img_path}: {str(e)}")
            # Clean up any partial files
            if Path(fp_path).exists():
                Path(fp_path).unlink()
            return []

    def _process_test_image(self, img_path: str) -> Dict:
        """
        Process a single test image.

        Args:
            img_path: Path to the test image

        Returns:
            Dict: Test results dictionary. Returns error info if processing fails.
        """
        try:
            return self._test_unrelated_image(img_path)
        except Exception as e:
            self.logger.error(f"Error processing test image {img_path}: {str(e)}")
            return {
                'scenario': 'unrelated',
                'test_image': img_path,
                'source_image': None,
                'verified': False,
                'similarity': 0.0,
                'modifications': [],
                'expected': False,
                'error': str(e)
            }

    def _test_original_image(self, fp_path: str, orig_path: str) -> Dict:
        """Test an original fingerprinted image."""
        try:
            # Verify ownership
            is_owned, orig_path_verified, similarity, mods = self.author.verify_ownership(fp_path)

            return {
                'scenario': 'original',
                'test_image': fp_path,
                'source_image': orig_path,
                'verified': is_owned,
                'similarity': similarity,
                'modifications': mods,
                'expected': True
            }
        except Exception as e:
            self.logger.error(f"Error testing original image {fp_path}: {str(e)}")
            raise

    def _test_manipulated_images(self, fp_path: str, orig_path: str) -> List[Dict]:
        """Test manipulated versions of a fingerprinted image."""
        results = []
        try:
            # Load fingerprinted image
            fp_img = cv2.imread(fp_path)
            if fp_img is None:
                raise ValueError(f"Could not read fingerprinted image: {fp_path}")

            # Test each manipulation type
            for manip_name in self.config['testing']['manipulations']:
                manipulated_img = self.manipulator.apply_manipulation(fp_img, manip_name)
                manip_path = str(self.run_manager.run_paths['manipulated'] / f"{manip_name}_{Path(fp_path).name}")
                cv2.imwrite(manip_path, manipulated_img)

                # Verify ownership of manipulated version
                is_owned, orig_path_found, similarity, mods = self.author.verify_ownership(manip_path)

                results.append({
                    'scenario': f'manipulated_{manip_name}',
                    'test_image': manip_path,
                    'source_image': orig_path,
                    'verified': is_owned,
                    'similarity': similarity,
                    'modifications': mods,
                    'expected': True
                })

        except Exception as e:
            self.logger.error(f"Error in manipulation test: {str(e)}")
            raise

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

    def _test_unrelated_image(self, img_path: str) -> Dict:
        """
        Test a single unrelated image.

        Args:
            img_path: Path to the test image

        Returns:
            Dict: Test results dictionary
        """
        if not Path(img_path).exists():
            error_msg = f"Image file not found: {img_path}"
            self.logger.error(error_msg)
            return {
                'scenario': 'unrelated',
                'test_image': img_path,
                'source_image': None,
                'verified': False,
                'similarity': 0.0,
                'modifications': [],
                'expected': False,
                'error': error_msg
            }

        try:
            # Verify ownership (should return False)
            is_owned, orig_path, similarity, mods = self.author.verify_ownership(img_path)

            return {
                'scenario': 'unrelated',
                'test_image': img_path,
                'source_image': None,
                'verified': is_owned,
                'similarity': similarity,
                'modifications': mods,
                'expected': False
            }

        except Exception as e:
            error_msg = f"Error testing unrelated image: {str(e)}"
            self.logger.error(error_msg)
            return {
                'scenario': 'unrelated',
                'test_image': img_path,
                'source_image': None,
                'verified': False,
                'similarity': 0.0,
                'modifications': [],
                'expected': False,
                'error': error_msg
            }

    def _generate_report(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
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
            'run_id': self.run_manager.run_id
        }

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

        # Calculate averages
        for data in report['by_scenario'].values():
            if data['total'] > 0:
                data['success_rate'] = (data['successful'] / data['total']) * 100
                data['average_similarity'] = data['average_similarity'] / data['total']

        return report

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save test report to files."""
        report_dir = self.run_manager.run_paths['reports']

        # Save detailed JSON report
        report_path = report_dir / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save summary
        summary_path = report_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            self._write_summary(f, report)

        self.logger.info(f"Saved test report to {report_path}")
        self.logger.info(f"Saved summary to {summary_path}")

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
