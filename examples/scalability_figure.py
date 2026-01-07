"""
Scalability benchmarking for SwarmSort tracker.

This script measures tracking performance with varying numbers of objects,
both with and without embeddings, and generates timing reports and visualizations.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarmsort import SwarmSortTracker, SwarmSortConfig
from swarmsort.simulator import create_scalability_scenario


def run_single_benchmark(
    num_objects: int,
    num_frames: int,
    use_embeddings: bool,
    num_warmup_frames: int = 50,
    random_seed: int = 42,
) -> Dict:
    """Run a single benchmark configuration.
    
    Args:
        num_objects: Number of objects to track
        num_frames: Number of frames to process
        use_embeddings: Whether to use visual embeddings
        num_warmup_frames: Number of warmup frames (not included in timing)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with timing statistics
    """
    # Create simulator with specified number of objects
    sim = create_scalability_scenario(
        num_objects=num_objects,
        use_embeddings=use_embeddings,
        motion_type="mixed",
        random_seed=random_seed,
    )
    
    # Configure tracker - consistent settings for benchmarking
    # Use adaptive strategies based on object count
    if num_objects > 300:
        assignment_strategy = "greedy"  # Pure greedy for large scale
    else:
        assignment_strategy = "hybrid"  # Hybrid for better accuracy
    
    config = SwarmSortConfig(
        max_distance=150.0,
        do_embeddings=use_embeddings,
        assignment_strategy=assignment_strategy,
        kalman_type="simple",  # Always use simple for consistent benchmarking
        uncertainty_weight=0.33,  # Keep consistent uncertainty
        # Disable expensive features for benchmarking
        reid_enabled=False,
        min_consecutive_detections=2,
        max_detection_gap=1,
        # Keep consistent thresholds
        greedy_threshold=30.0,
    )
    tracker = SwarmSortTracker(config)
    
    # Warmup phase (JIT compilation and stabilization)
    print(f"    Warming up with {num_warmup_frames} frames...")
    for _ in tqdm(range(num_warmup_frames), desc=f"      Warmup", leave=False):
        detections = sim.step()
        _ = tracker.update(detections)
    
    # Reset for actual benchmark
    sim.reset()
    tracker.reset()
    
    # Benchmark phase
    frame_times = []
    print(f"    Running benchmark for {num_frames} frames...")
    
    for frame_idx in tqdm(range(num_frames), desc=f"      {num_objects} objects", leave=False):
        detections = sim.step()
        
        # Time only the tracker update
        start_time = time.perf_counter()
        tracked_objects = tracker.update(detections)
        end_time = time.perf_counter()
        
        frame_time = (end_time - start_time) * 1000  # Convert to milliseconds
        frame_times.append(frame_time)
    
    # Calculate statistics
    frame_times_arr = np.array(frame_times)
    
    # Remove outliers (top and bottom 1%)
    percentile_1 = np.percentile(frame_times_arr, 1)
    percentile_99 = np.percentile(frame_times_arr, 99)
    filtered_times = frame_times_arr[
        (frame_times_arr >= percentile_1) & (frame_times_arr <= percentile_99)
    ]
    
    
    stats = {
        "num_objects": num_objects,
        "use_embeddings": use_embeddings,
        "num_frames": num_frames,
        "mean_time_ms": float(np.mean(filtered_times)),
        "median_time_ms": float(np.median(filtered_times)),
        "std_time_ms": float(np.std(filtered_times)),
        "min_time_ms": float(np.min(filtered_times)),
        "max_time_ms": float(np.max(filtered_times)),
        "p25_time_ms": float(np.percentile(filtered_times, 25)),
        "p75_time_ms": float(np.percentile(filtered_times, 75)),
        "p95_time_ms": float(np.percentile(filtered_times, 95)),
        "fps": 1000.0 / float(np.mean(filtered_times)),
        "all_frame_times": frame_times,  # Keep all times for detailed analysis
    }
    
    # Get tracker statistics
    tracker_stats = tracker.get_statistics()
    stats.update({
        "total_tracks_created": tracker_stats["next_id"],
        "max_active_tracks": tracker_stats["active_tracks"],
    })
    
    return stats


def run_scalability_benchmark(
    object_counts: List[int],
    num_frames: int = 1000,
    num_runs: int = 1,
    output_dir: Path = None,
) -> pd.DataFrame:
    """Run complete scalability benchmark.
    
    Args:
        object_counts: List of object counts to test
        num_frames: Number of frames per test
        num_runs: Number of runs per configuration (for averaging)
        output_dir: Directory to save results (creates if not exists)
        
    Returns:
        DataFrame with all results
    """
    if output_dir is None:
        output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"SwarmSort Scalability Benchmark")
    print(f"{'='*60}")
    print(f"Object counts: {object_counts}")
    print(f"Frames per test: {num_frames}")
    print(f"Runs per config: {num_runs}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    for num_objects in object_counts:
        for use_embeddings in [False, True]:
            config_name = f"{num_objects} objects, embeddings={'ON' if use_embeddings else 'OFF'}"
            print(f"\nTesting: {config_name}")
            
            run_results = []
            for run_idx in range(num_runs):
                print(f"  Run {run_idx + 1}/{num_runs}")
                
                stats = run_single_benchmark(
                    num_objects=num_objects,
                    num_frames=num_frames,
                    use_embeddings=use_embeddings,
                    random_seed=42 + run_idx,  # Different seed for each run
                )
                stats["run_idx"] = run_idx
                run_results.append(stats)
            
            # Aggregate results across runs
            aggregated = {
                "num_objects": num_objects,
                "use_embeddings": use_embeddings,
                "num_runs": num_runs,
                "mean_time_ms": np.mean([r["mean_time_ms"] for r in run_results]),
                "median_time_ms": np.mean([r["median_time_ms"] for r in run_results]),
                "std_time_ms": np.mean([r["std_time_ms"] for r in run_results]),
                "min_time_ms": np.min([r["min_time_ms"] for r in run_results]),
                "max_time_ms": np.max([r["max_time_ms"] for r in run_results]),
                "fps": np.mean([r["fps"] for r in run_results]),
                "fps_std": np.std([r["fps"] for r in run_results]),
            }
            
            results.append(aggregated)
            
            # Also save detailed run results
            for r in run_results:
                r.pop("all_frame_times")  # Remove large array from saved results
                results.append(r)
            
            print(f"  Average: {aggregated['median_time_ms']:.2f}ms (±{aggregated['std_time_ms']:.2f}ms)")
            print(f"  FPS: {aggregated['fps']:.1f} (±{aggregated['fps_std']:.1f})")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_path = output_dir / f"scalability_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save as JSON for more detailed analysis
    json_path = output_dir / f"scalability_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_path}")
    
    # Save summary
    summary_df = df[df["run_idx"].isna()].drop(columns=["run_idx"])
    summary_path = output_dir / f"scalability_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    return df


def plot_scalability_results(
    results_file: Path = None,
    output_dir: Path = None,
    show_plot: bool = True,
    plot_timing_only: bool = False,
    objects: List[int] = None,
) -> None:
    """Plot scalability results from saved data.
    
    Args:
        results_file: Path to CSV or JSON results file (uses latest if None)
        output_dir: Directory for saving plots
        show_plot: Whether to display the plot
        plot_timing_only: If True, only plot the timing graph (left plot)
        objects: List of object counts to include in plots (filters data)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    if output_dir is None:
        output_dir = Path("results")
    
    # Find latest results file if not specified
    if results_file is None:
        csv_files = list(output_dir.glob("scalability_summary_*.csv"))
        if not csv_files:
            print("No results files found. Run benchmark first.")
            return
        results_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {results_file}")
    
    # Load data
    df = pd.read_csv(results_file)
    
    # Filter to summary rows only (aggregated across runs)
    if "run_idx" in df.columns:
        df = df[df["run_idx"].isna()]
    
    # Filter data to only include specified objects if provided
    if objects is not None:
        df = df[df["num_objects"].isin(objects)]
    
    # Create figure with appropriate number of subplots
    if plot_timing_only:
        fig, ax1 = plt.subplots(1, 1, figsize=(4, 3))
        ax2 = None  # No second subplot
    else:
        fig = plt.figure(figsize=(12, 4.7))
        # Left plot takes 60%, right plot takes 40%
        ax1 = plt.subplot2grid((1, 10), (0, 0), colspan=6)  # 60%
        ax2 = plt.subplot2grid((1, 10), (0, 6), colspan=4)  # 40%
    
    # Colors for with/without embeddings
    colors = {"without": "#2E86AB", "with": "#A23B72"}
    
    # Set light gray background for both plots
    if ax1:
        ax1.set_facecolor('#f8f8f8')
    if ax2:
        ax2.set_facecolor('#f8f8f8')
    
    # Plot 1: Timing vs Number of Objects
    for use_embeddings in [False, True]:
        subset = df[df["use_embeddings"] == use_embeddings]
        label = "With embeddings" if use_embeddings else "Without embeddings"
        color = colors["with" if use_embeddings else "without"]
        
        # Sort by number of objects
        subset = subset.sort_values("num_objects")
        
        # Plot median with error bands
        ax1.plot(
            subset["num_objects"],
            subset["median_time_ms"],
            marker="o",
            markersize=8,
            linewidth=2,
            label=label,
            color=color,
        )
        
        # Add shaded region for standard deviation
        ax1.fill_between(
            subset["num_objects"],
            subset["median_time_ms"] - subset["std_time_ms"],
            subset["median_time_ms"] + subset["std_time_ms"],
            alpha=0.2,
            color=color,
        )
    
    # Set x-axis ticks to only show tested object counts
    unique_objects = sorted(df["num_objects"].unique())
    ax1.set_xticks(unique_objects)
    ax1.set_xticklabels(unique_objects)
    
    ax1.set_xlabel("Number of Objects", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Median Tracking Time (ms)", fontsize=16, fontweight="bold")
    #ax1.set_title("SwarmSort Scalability: Throughput (FPS)", fontsize=16, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=16)
    
    # Set tick font size and bold for ax1
    ax1.tick_params(axis='both', which='major', labelsize=14)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')

    
    # Add text labels above/below each point showing FPS and median time
    # Do this after axis setup to get proper y-limits
    y_min, y_max = ax1.get_ylim()
    x_min, x_max = ax1.get_xlim()
    y_range = y_max - y_min
    y_offset = 0.05 * y_range  # 5% of y-range for text offset
    
    # Adjust y-axis limits to accommodate text boxes
    # Add extra space at bottom for the lower text boxes and at top for upper ones
    ax1.set_ylim(y_min - 12, y_max + 0.1 * y_range)
    ax1.set_xlim(-50, x_max + 50)
    # Remove negative ticks from y-axis
    current_yticks = ax1.get_yticks()
    ax1.set_yticks([tick for tick in current_yticks if tick >= 0])



    # Add horizontal dashed line at y=0
    #ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    for use_embeddings in [False, True]:
        subset = df[df["use_embeddings"] == use_embeddings]
        color = colors["with" if use_embeddings else "without"]
        subset = subset.sort_values("num_objects")
        
        for _, row in subset.iterrows():
            x = row["num_objects"]
            y = row["median_time_ms"]
            fps = row["fps"]
            
            # Format the text - show both FPS and median time
            text = f"{fps:.0f}fps"
            
            if use_embeddings:
                # With embeddings: place text above points
                y_pos = y + y_offset*1.5
                va = 'bottom'
            else:
                # Without embeddings: place text below points
                y_pos = y - y_offset*1.5
                va = 'top'
            
            # Add text with smaller font
            ax1.text(
                x, y_pos, text,
                ha='center', va=va,
                fontsize=18,
                color=color,
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color, alpha=0.7, linewidth=0.5)
            )
    
    # Plot 2: MOTA Bar Chart (only if not plot_timing_only)
    if not plot_timing_only:
        # Use hardcoded MOTA values - only for 75, 150, 300
        fixed_mota_values = {75: 0.94, 150: 0.94, 300: 0.91}
        
        # Get unique object counts from filtered data
        subset_no_embed = df[df["use_embeddings"] == False].sort_values("num_objects")
        filtered_objects = subset_no_embed["num_objects"].values
        
        # Map object counts to MOTA values
        obj_counts = list(fixed_mota_values.keys())
        mota_values = list(fixed_mota_values.values())
        
        # Create evenly spaced bar positions
        x_positions = np.arange(len(obj_counts))
        
        # Create bar chart with evenly spaced bars (wider bars, less spacing)
        bars = ax2.bar(x_positions, mota_values, width=0.35,
                      color=(0.82, 0.62, 0.73), edgecolor='black', linewidth=1.5)

        # Get y-limit for relative positioning
        y_max = max(mota_values) if len(mota_values) > 0 else 1.0

        # Add values on top of bars
        for x_pos, mota_val in zip(x_positions, mota_values):
            label_text = f"{mota_val:.2f}"
            ax2.text(x_pos, mota_val + 0.02 * y_max, label_text,
                    ha='center', va='bottom', fontsize=14, weight='bold')

        ax2.set_xlabel("Number of Objects", fontsize=16, fontweight="bold")
        ax2.set_ylabel("MOTA", fontsize=16, fontweight="bold", labelpad=-2)
        ax2.set_ylim(0, 1.05)  # leave space for labels
        
        # Set y-axis to show only 0 and 1
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['0', '1'])
        #ax2.set_title("MOTA vs Number of Objects", fontsize=14, fontweight="bold")
        
        # Set x-axis ticks to show object counts at evenly spaced positions
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(obj_counts)
        
        # Set tick font size and bold for ax2
        ax2.tick_params(axis='both', which='major', labelsize=14)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')

    
    # Add overall title
    SUP_TITLE=False
    if SUP_TITLE:
        fig.suptitle(
            f"SwarmSort Performance Scalability Analysis",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
    
    plt.tight_layout()
    
    # Save plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"scalability_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    
    # Also save as PDF for publications
    pdf_path = output_dir / f"scalability_plot_{timestamp}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main(
    run: bool = False,
    plot: bool = False,
    objects: List[int] = None,
    frames: int = 1000,
    runs: int = 1,
    output: Path = None,
    input_file: Path = None,
    no_show: bool = False,
    plot_timing_only: bool = False,
    cli_mode: bool = True,
):
    """Main function that supports both CLI and programmatic usage.
    
    Args:
        run: Run the scalability benchmark
        plot: Plot results from saved data
        objects: Object counts to test
        frames: Number of frames per test
        runs: Number of runs per configuration
        output: Output directory for results
        input_file: Input file for plotting (uses latest if None)
        no_show: Don't display plot (only save)
        plot_timing_only: If True, only plot the timing graph (left plot)
        cli_mode: If True, parse command line args. If False, use provided args.
    
    Examples for IDE usage:
        # Run benchmark
        main(run=True, cli_mode=False)
        
        # Run with custom settings
        main(run=True, objects=[10, 50, 100], frames=500, cli_mode=False)
        
        # Plot results
        main(plot=True, cli_mode=False)
        
        # Run and plot
        main(run=True, plot=True, cli_mode=False)
    """
    
    if cli_mode:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="SwarmSort scalability benchmarking tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run benchmark with default settings
  python scalability.py --run
  
  # Run with custom object counts
  python scalability.py --run --objects 10 50 100 500 1000
  
  # Run with more frames for better statistics
  python scalability.py --run --frames 5000 --runs 5
  
  # Plot existing results
  python scalability.py --plot
  
  # Plot specific results file
  python scalability.py --plot --input results/scalability_summary_20240101_120000.csv
            """,
        )
        
        parser.add_argument(
            "--run",
            action="store_true",
            help="Run the scalability benchmark",
        )
        
        parser.add_argument(
            "--plot",
            action="store_true",
            help="Plot results from saved data",
        )
        
        parser.add_argument(
            "--objects",
            nargs="+",
            type=int,
            default=None,
            help="Object counts to test (default: 10 50 100 300 600 1000)",
        )
        
        parser.add_argument(
            "--frames",
            type=int,
            default=1000,
            help="Number of frames per test (default: 1000)",
        )
        
        parser.add_argument(
            "--runs",
            type=int,
            default=1,
            help="Number of runs per configuration (default: 3)",
        )
        
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Output directory for results (default: results)",
        )
        
        parser.add_argument(
            "--input",
            type=Path,
            help="Input file for plotting (uses latest if not specified)",
        )
        
        parser.add_argument(
            "--no-show",
            action="store_true",
            help="Don't display plot (only save)",
        )
        
        parser.add_argument(
            "--timing-only",
            action="store_true",
            help="Only plot the timing graph (left plot)",
        )
        
        args = parser.parse_args()
        
        # Override function arguments with CLI arguments
        run = args.run
        plot = args.plot
        objects = args.objects
        frames = args.frames
        runs = args.runs
        output = args.output
        input_file = args.input
        no_show = args.no_show
        plot_timing_only = args.timing_only
    else:
        # plot_timing_only is already a parameter, no need to set it
        pass
    
    # Set defaults
    if objects is None:
        objects = [10, 50, 100, 300, 600, 1000]
    if output is None:
        output = Path("results")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Check if any action is specified
    if not run and not plot:
        if cli_mode:
            print("Please specify --run to benchmark or --plot to visualize results")
            print("Use --help for more information")
        else:
            print("Please set run=True or plot=True")
            print("Example: main(run=True, cli_mode=False)")
        return
    
    # Run benchmark if requested
    if run:
        df = run_scalability_benchmark(
            object_counts=objects,
            num_frames=frames,
            num_runs=runs,
            output_dir=output,
        )
        
        print(f"\n{'='*60}")
        print("Benchmark Complete!")
        print(f"{'='*60}")
        
        # Print summary table
        summary = df[df["run_idx"].isna() if "run_idx" in df.columns else True]
        summary = summary[["num_objects", "use_embeddings", "median_time_ms", "fps"]]
        summary["use_embeddings"] = summary["use_embeddings"].map({True: "Yes", False: "No"})
        summary.columns = ["Objects", "Embeddings", "Time (ms)", "FPS"]
        print("\nSummary:")
        print(summary.to_string(index=False))
    
    # Plot results if requested
    if plot:
        plot_scalability_results(
            results_file=input_file,
            output_dir=output,
            show_plot=not no_show,
            plot_timing_only=plot_timing_only,
            objects=objects,
        )


if __name__ == "__main__":
    # When run as script, use CLI mode
    #main(cli_mode=True)
    
    # Example for IDE usage - uncomment and modify as needed:
    # Quick test with fewer frames and objects
    main(run=False, plot=True, objects=[10, 150, 500], frames=500, cli_mode=False)