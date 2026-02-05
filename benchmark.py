import subprocess
import re
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def run_simulation(
    executable, n_particles, block_size, mode, iterations=10, steps_per_frame=1
):
    cmd = [
        executable,
        "-n",
        str(n_particles),
        "-b",
        str(block_size),
        "-m",
        mode,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,  # skip the csv
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=300,
        )

        match = re.search(r"Total elapsed \(s\):\s+([\d.]+)", result.stderr)
        if match:
            # print (cmd, float(match.group(1)))
            return float(match.group(1))
        else:
            print(f"Warning: Could not parse timing from output: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout for n={n_particles}, block_size={block_size}, mode={mode}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def run_benchmarks(executable="./nbody_sim"):
    n_particles_list = [512, 1024, 2048, 4096, 8192, 16384]
    block_sizes = [64, 128, 256, 512]
    modes = ["naive", "tiled", "tree"]
    iterations = 10
    steps_per_frame = 1

    results = {}

    total_runs = len(n_particles_list) * len(block_sizes) * len(modes)
    current_run = 0

    print(f"\nRunning {total_runs} benchmark configurations...\n")

    for block_size in block_sizes:
        results[block_size] = {}

        for mode in modes:
            results[block_size][mode] = {}

            for n_particles in n_particles_list:
                current_run += 1
                print(
                    f"[{current_run}/{total_runs}] Running: n={n_particles}, block_size={block_size}, mode={mode}...",
                    end=" ",
                )

                time_seconds = run_simulation(
                    executable,
                    n_particles,
                    block_size,
                    mode,
                    iterations,
                    steps_per_frame,
                )

                if time_seconds is not None:
                    results[block_size][mode][n_particles] = time_seconds
                    print(f"{time_seconds:.4f}s")
                else:
                    print("FAILED")
                    results[block_size][mode][n_particles] = None

    return results


def plot_results(results, output_dir="benchmark_plots"):
    Path(output_dir).mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")

    for block_size, modes_data in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for mode, n_data in modes_data.items():
            n_particles_list = sorted(n_data.keys())
            times = [n_data[n] for n in n_particles_list if n_data[n] is not None]
            valid_n = [n for n in n_particles_list if n_data[n] is not None]

            if times:
                ax.plot(
                    valid_n,
                    times,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=mode.capitalize(),
                )

        ax.set_xlabel("Number of Particles", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title(
            f"N-Body Simulation Performance (Block Size: {block_size})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # better vis
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        plt.tight_layout()

        output_file = os.path.join(output_dir, f"performance_block_{block_size}.png")
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot: {output_file}")

        plt.close()


def print_summary(results):
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    for block_size, modes_data in sorted(results.items()):
        print(f"\nBlock Size: {block_size}")
        print("-" * 80)
        print(
            f"{'N Particles':<15} {'Naive (s)':<15} {'Tiled (s)':<15} {'Speedup':<15}"
        )
        print("-" * 80)

        if "naive" in modes_data and "tiled" in modes_data:
            naive_data = modes_data["naive"]
            tiled_data = modes_data["tiled"]

            n_particles_list = sorted(set(naive_data.keys()) | set(tiled_data.keys()))

            for n in n_particles_list:
                naive_time = naive_data.get(n, None)
                tiled_time = tiled_data.get(n, None)

                naive_str = f"{naive_time:.4f}" if naive_time else "N/A"
                tiled_str = f"{tiled_time:.4f}" if tiled_time else "N/A"

                if naive_time and tiled_time and tiled_time > 0:
                    speedup = naive_time / tiled_time
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"

                print(f"{n:<15} {naive_str:<15} {tiled_str:<15} {speedup_str:<15}")


if __name__ == "__main__":
    print("=" * 80)
    print("CUDA N-Body Simulation Benchmark")
    print("=" * 80)

    executable = "./nbody_sim"

    if not os.path.exists(executable):
        print(f"Error: Executable '{executable}' still not found after compilation.")
        sys.exit(1)

    results = run_benchmarks(executable)

    print_summary(results)

    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    plot_results(results)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)