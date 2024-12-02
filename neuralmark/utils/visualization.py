import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
import logging
from pathlib import Path


def setup_plot_style():
    """Setup consistent plot style."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14


def visualize_attention_maps(image: np.ndarray,
                             attention_maps: Dict[str, List[np.ndarray]],
                             save_path: Optional[str] = None,
                             config: Optional[dict] = None) -> None:
    """Visualize attention maps from different algorithms."""
    if config is None:
        config = {
            'visualization': {
                'attention_maps': {
                    'colormap': 'viridis',
                    'alpha': 0.7
                },
                'dpi': 300
            }
        }

    setup_plot_style()

    # Calculate total number of maps across all algorithms
    total_maps = sum(len(maps) for maps in attention_maps.values())
    fig, axes = plt.subplots(len(attention_maps), max(len(maps) for maps in attention_maps.values()) + 1,
                             figsize=(
                                 5 * (max(len(maps) for maps in attention_maps.values()) + 1), 5 * len(attention_maps)))

    # Ensure axes is always a 2D array
    if len(attention_maps) == 1:
        axes = np.array([axes])

    for row, (algo, maps) in enumerate(attention_maps.items()):
        # Plot original image in first column
        axes[row, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f'{algo.upper()} - Original')
        axes[row, 0].axis('off')

        # Plot attention maps
        for i, attention_map in enumerate(maps):
            # Normalize attention map
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
            norm_map = cv2.normalize(attention_map, None, 0, 1, cv2.NORM_MINMAX)

            # Apply colormap
            colormap = config['visualization']['attention_maps'].get('colormap', 'viridis')
            alpha = config['visualization']['attention_maps'].get('alpha', 0.7)

            axes[row, i + 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            im = axes[row, i + 1].imshow(norm_map, cmap=colormap, alpha=alpha)
            axes[row, i + 1].set_title(f'{algo.upper()} - Attention {i + 1}')
            axes[row, i + 1].axis('off')
            plt.colorbar(im, ax=axes[row, i + 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config['visualization'].get('dpi', 300))
        plt.close()
    else:
        plt.show()


def plot_algorithm_comparison(results: Dict[str, Dict],
                              metrics: List[str],
                              save_path: Optional[str] = None) -> None:
    """Plot comparison of different algorithms."""
    setup_plot_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        values = [result['summary'][metric] for result in results.values()]
        algorithms = list(results.keys())

        axes[i].bar(algorithms, values, color=['royalblue', 'forestgreen'])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_test_results(results: Dict[str, Dict], run_dir: Path, logger: logging.Logger) -> None:
    """Generate comprehensive visualization of test results."""
    plots_dir = run_dir / 'plots'

    try:
        for algo, result in results.items():
            # Create algorithm-specific directory if needed
            algo_dir = plots_dir / algo if len(results) > 1 else plots_dir
            algo_dir.mkdir(parents=True, exist_ok=True)

            # 1. Success rates by scenario
            plt.figure(figsize=(12, 6))
            scenarios = list(result['by_scenario'].keys())
            success_rates = [data['success_rate'] for data in result['by_scenario'].values()]

            plt.bar(scenarios, success_rates)
            plt.title(f'{algo.upper()} - Success Rates by Scenario')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Success Rate (%)')
            plt.tight_layout()
            plt.savefig(algo_dir / 'success_rates.png')
            plt.close()

            # 2. Similarity distributions
            plt.figure(figsize=(12, 6))
            similarities = [case['similarity'] for case in result['test_cases']]
            plt.hist(similarities, bins=30)
            plt.title(f'{algo.upper()} - Distribution of Similarity Scores')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(algo_dir / 'similarity_distribution.png')
            plt.close()

            # 3. Modification detection accuracy
            mods_detected = {}
            for case in result['test_cases']:
                for mod in case.get('modifications', []):
                    mods_detected[mod] = mods_detected.get(mod, 0) + 1

            if mods_detected:
                plt.figure(figsize=(12, 6))
                plt.bar(list(mods_detected.keys()), list(mods_detected.values()))
                plt.title(f'{algo.upper()} - Detected Modifications')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(algo_dir / 'modifications.png')
                plt.close()

            # 4. Error timeline
            plt.figure(figsize=(12, 6))
            test_indices = range(len(result['test_cases']))
            errors = [1 if case['verified'] != case['expected'] else 0
                      for case in result['test_cases']]
            plt.plot(test_indices, errors, 'r.')
            plt.title(f'{algo.upper()} - Errors Over Time')
            plt.xlabel('Test Case Index')
            plt.ylabel('Error (0=correct, 1=error)')
            plt.tight_layout()
            plt.savefig(algo_dir / 'error_timeline.png')
            plt.close()

        # Generate comparison plots if multiple algorithms
        if len(results) > 1:
            plot_algorithm_comparison(
                results,
                ['success_rate', 'false_positives', 'false_negatives'],
                plots_dir / 'algorithm_comparison.png'
            )

        logger.info(f"Generated result visualizations in {plots_dir}")

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")


def save_visualization(plot, name: str, run_dir: Path, logger: logging.Logger, algorithm: str = None):
    """Save visualization with proper directory structure.

    Args:
        plot: Matplotlib plot object
        name: Name of the plot file
        run_dir: Run directory path
        logger: Logger instance
        algorithm: Optional algorithm name for subdirectory
    """
    plots_dir = run_dir / 'plots'
    if algorithm:
        plots_dir = plots_dir / algorithm

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{name}.png"

    try:
        plot.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot.close()
    except Exception as e:
        logger.error(f"Error saving plot {name}: {str(e)}")


def create_comparison_grid(images: List[Tuple[str, str, np.ndarray]],
                           save_path: Optional[str] = None,
                           title: str = "Algorithm Comparison") -> None:
    """Create a grid comparing results from different algorithms."""
    setup_plot_style()

    n_images = len(images)
    if n_images == 0:
        return

    # Group images by algorithm
    algo_groups = {}
    for algo, label, img in images:
        if algo not in algo_groups:
            algo_groups[algo] = []
        algo_groups[algo].append((label, img))

    # Calculate grid dimensions
    n_algos = len(algo_groups)
    n_samples = max(len(group) for group in algo_groups.values())

    fig, axes = plt.subplots(n_algos, n_samples, figsize=(4 * n_samples, 4 * n_algos))

    if n_algos == 1:
        axes = [axes]

    for i, (algo, group) in enumerate(algo_groups.items()):
        for j, (label, img) in enumerate(group):
            if img.ndim == 2:
                axes[i][j].imshow(img, cmap='gray')
            else:
                axes[i][j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            axes[i][j].set_title(f'{algo.upper()} - {label}')
            axes[i][j].axis('off')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_modification_analysis(originals: Dict[str, np.ndarray],
                               modifieds: Dict[str, np.ndarray],
                               differences: Dict[str, np.ndarray],
                               save_path: Optional[str] = None) -> None:
    """Visualize original, modified, and difference images for each algorithm."""
    setup_plot_style()

    n_algos = len(originals)
    fig, axes = plt.subplots(n_algos, 3, figsize=(15, 5 * n_algos))

    if n_algos == 1:
        axes = [axes]

    for i, algo in enumerate(originals.keys()):
        # Plot original
        axes[i][0].imshow(cv2.cvtColor(originals[algo], cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f'{algo.upper()} - Original')
        axes[i][0].axis('off')

        # Plot modified
        axes[i][1].imshow(cv2.cvtColor(modifieds[algo], cv2.COLOR_BGR2RGB))
        axes[i][1].set_title(f'{algo.upper()} - Modified')
        axes[i][1].axis('off')

        # Plot differences
        norm_diff = cv2.normalize(differences[algo], None, 0, 1, cv2.NORM_MINMAX)
        im = axes[i][2].imshow(norm_diff, cmap='hot')
        axes[i][2].set_title(f'{algo.upper()} - Differences')
        axes[i][2].axis('off')
        plt.colorbar(im, ax=axes[i][2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
