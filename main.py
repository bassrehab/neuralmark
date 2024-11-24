import argparse
import logging
from pathlib import Path

from cleanup import DirectoryCleaner
from test_authorship import AuthorshipTester
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description='AlphaPunch Image Fingerprinting System')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['test', 'benchmark', 'cross_validation'],
                        default='test', help='Operation mode')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for results')
    args = parser.parse_args()

    # Create cleaner
    cleaner = DirectoryCleaner()

    try:
        # Load configuration
        config = load_config(args.config)

        # Setup output directory
        Path(args.output).mkdir(parents=True, exist_ok=True)

        # Pre-run cleanup
        cleaner.cleanup(pre_run=True)

        # Initialize tester
        tester = AuthorshipTester(config_path=args.config)

        # Run based on mode
        if args.mode == 'test':
            results = tester.run_authorship_tests()
            print("\nTest Results Summary:")
            print(f"Total Tests: {results['summary']['total_tests']}")
            print(f"Successful Verifications: {results['summary']['successful_verifications']}")
            print(f"False Positives: {results['summary']['false_positives']}")
            print(f"False Negatives: {results['summary']['false_negatives']}")

        elif args.mode == 'benchmark':
            results = tester.run_authorship_tests()  # This includes performance metrics
            print("\nBenchmark Results:")
            print(f"Average Processing Time: {results['performance_metrics']['execution_times']['mean']:.3f}s")
            print(f"Peak Memory Usage: {results['performance_metrics']['memory_usage']['peak']:.2f}MB")

        elif args.mode == 'cross_validation':
            results = tester.run_cross_validation()
            print("\nCross-Validation Results:")
            print(f"Average Success Rate: {results['average_success_rate']:.2f}%")
            print(f"Standard Deviation: {results['std_success_rate']:.2f}%")

            # Post-run cleanup
            cleaner.cleanup(pre_run=False)

    except Exception as e:
        logging.error(f"Error running program: {str(e)}")
        raise


if __name__ == "__main__":
    main()
