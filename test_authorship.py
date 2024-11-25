import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from alphapunch.author import ImageAuthor
from utils import load_config, setup_logger, get_test_images, ImageManipulator


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

    def run_authorship_tests(self) -> Dict[str, Any]:
        """Run complete authorship verification test suite."""
        self.logger.info(f"{__name__}.run_authorship_tests - Starting authorship verification tests...")

        test_cases = []
        start_time = time.time()

        try:
            # 1. Get test dataset
            total_images = self.config['testing']['total_images']
            train_ratio = self.config['testing']['train_ratio']
            train_size = int(total_images * train_ratio)
            test_size = total_images - train_size

            all_images = get_test_images(total_images, self.config, self.logger)
            train_images = all_images[:train_size]
            test_images = all_images[train_size:]

            # 2. Process training images
            self.logger.info("Testing with original images...")
            fingerprinted_pairs = []

            for img_path in tqdm(train_images, desc="Testing originals"):
                try:
                    # Fingerprint and save
                    fp_path = str(self.run_manager.run_paths['fingerprinted'] / Path(img_path).name)
                    fingerprinted_img, fingerprint = self.author.fingerprint_image(img_path, fp_path)
                    fingerprinted_pairs.append((img_path, fp_path))

                    # Test original fingerprinted image
                    result = self._test_original_image(fp_path, img_path)
                    test_cases.extend(result)

                except Exception as e:
                    self.logger.error(f"Error testing original image {img_path}: {str(e)}")
                    continue

            # 3. Test manipulated versions
            self.logger.info("Testing with manipulated images...")
            manipulated_results = []

            for orig_path, fp_path in fingerprinted_pairs:
                try:
                    results = self._test_manipulated_images(fp_path, orig_path)
                    manipulated_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Error testing manipulated image {fp_path}: {str(e)}")
                    continue

            test_cases.extend(manipulated_results)

            # 4. Test unrelated images
            self.logger.info("Testing with unrelated images...")
            unrelated_results = self._test_unrelated_images(test_images)
            test_cases.extend(unrelated_results)

            # 5. Generate and save report
            report = self._generate_report(test_cases)
            report['performance'] = {
                'execution_time': time.time() - start_time,
                'dataset': {
                    'total_images': total_images,
                    'train_images': train_size,
                    'test_images': test_size
                }
            }

            self._save_report(report)
            self._generate_visualizations(report)

            return report

        except Exception as e:
            self.logger.error(f"Error during authorship tests: {str(e)}")
            raise

    def _test_original_image(self, fp_path: str, orig_path: str) -> List[Dict]:
        """Test an original fingerprinted image."""
        results = []

        # Verify ownership
        is_owned, orig_path, similarity, mods = self.author.verify_ownership(fp_path)

        results.append({
            'scenario': 'original',
            'test_image': fp_path,
            'source_image': orig_path,
            'verified': is_owned,
            'similarity': similarity,
            'modifications': mods,
            'expected': True
        })

        return results

    def _test_manipulated_images(self, fp_path: str, orig_path: str) -> List[Dict]:
        """Test manipulated versions of a fingerprinted image."""
        results = []
        try:
            # Load fingerprinted image
            fp_img = cv2.imread(fp_path)
            if fp_img is None:
                raise ValueError(f"Could not read fingerprinted image: {fp_path}")

            # Apply each manipulation and test
            for manip_name in self.config['testing']['manipulations']:
                # Create manipulated version
                manipulated_img = self.manipulator.apply_manipulation(fp_img, manip_name)
                manip_path = str(self.run_manager.run_paths['manipulated'] / f"{manip_name}_{Path(fp_path).name}")
                cv2.imwrite(manip_path, manipulated_img)

                # Verify ownership
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
        """Test unrelated images (should not be recognized)."""
        results = []

        for img_path in tqdm(test_images, desc="Testing unrelated"):
            try:
                # Verify ownership (should fail)
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
                self.logger.error(f"Error testing unrelated image {img_path}: {str(e)}")
                continue

        return results

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

        # Set style
        plt.style.use('seaborn')

        # 1. Success rates by scenario
        self._plot_success_rates(report, plots_dir)

        # 2. Similarity distributions
        self._plot_similarity_distributions(report, plots_dir)

        # 3. Modification types
        self._plot_modification_types(report, plots_dir)

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