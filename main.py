import argparse
import logging
from pathlib import Path

from cleanup import DirectoryCleaner
from test_authorship import AuthorshipTester
from neuralmark.utils import load_config, setup_logger


def main():
    parser = argparse.ArgumentParser(description='NeuralMark Image Fingerprinting System')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['test', 'benchmark', 'cross_validation'],
                      help='Operation mode')
    parser.add_argument('--algorithm', type=str,
                      choices=['amdf', 'cdha', 'both'],
                      help='Algorithm to use (overrides config setting)')
    parser.add_argument('--output', type=str, default='output',
                      help='Output directory for results')
    parser.add_argument('--log-level', type=str,
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO', help='Set logging level')
    parser.add_argument('--no-cleanup', action='store_true',
                      help='Disable cleanup operations')
    parser.add_argument('--preserve-runs', type=int,
                      help='Number of previous runs to preserve')

    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger('NeuralMark')

    try:
        # Load configuration
        config = load_config(args.config)

        # Override settings from command line
        if args.algorithm:
            if args.algorithm == 'both':
                config['algorithm_selection']['enable_comparison'] = True
                logger.info("Enabled comparison mode for both algorithms")
            else:
                config['algorithm_selection']['type'] = args.algorithm
                config['algorithm_selection']['enable_comparison'] = False
                logger.info(f"Using {args.algorithm.upper()} algorithm")

        if args.no_cleanup:
            config['cleanup']['enabled'] = False
        if args.preserve_runs is not None:
            config['cleanup']['preserve_runs'] = args.preserve_runs

        # Setup output directory
        Path(args.output).mkdir(parents=True, exist_ok=True)

        # Handle cleanup if enabled
        if config['cleanup']['enabled']:
            cleaner = DirectoryCleaner(config_path=args.config)
            if config['cleanup']['pre_run']:
                cleaner.cleanup(pre_run=True)

        # Initialize tester
        tester = AuthorshipTester(config_path=args.config)

        # Run based on mode
        if args.mode == 'test':
            results = tester.run_authorship_tests()
            print("\nTest Results:")
            for algo, result in results.items():
                print(f"\n{algo.upper()}:")
                print(f"Total Tests: {result['summary']['total_tests']}")
                print(f"Success Rate: {result['summary']['success_rate']:.2f}%")
                print(f"False Positives: {result['summary']['false_positives']}")
                print(f"False Negatives: {result['summary']['false_negatives']}")

        elif args.mode == 'benchmark':
            results = tester.run_authorship_tests()
            print("\nBenchmark Results:")
            for algo, result in results.items():
                print(f"\n{algo.upper()} Performance:")
                print(f"Average Processing Time: {result['performance']['execution_time']:.3f}s")
                if 'memory_usage' in result['performance']:
                    print(f"Peak Memory Usage: {result['performance']['memory_usage']['peak']:.2f}MB")

        elif args.mode == 'cross_validation':
            results = tester.run_cross_validation()
            print("\nCross-Validation Results:")
            for algo, result in results.items():
                if algo != 'comparison':
                    print(f"\n{algo.upper()}:")
                    print(f"Average Success Rate: {result['average_success_rate']:.2f}%")
                    print(f"Standard Deviation: {result['std_success_rate']:.2f}%")
                    print(f"Min Success Rate: {result['min_success_rate']:.2f}%")
                    print(f"Max Success Rate: {result['max_success_rate']:.2f}%")

        # Handle post-run cleanup if enabled
        if config['cleanup']['enabled'] and config['cleanup']['post_run']:
            cleaner.cleanup(pre_run=False)

        logger.info("Program completed successfully")

    except Exception as e:
        logger.error(f"Error running program: {str(e)}")
        raise


if __name__ == "__main__":
    main()
