# 🌌 CUDA N-Body Galaxy Simulation

## 1. Project Overview

This project presents a GPU-accelerated N-body gravitational simulation implemented in CUDA. The goal is to model the motion of a large number of particles interacting under gravity, forming a galaxy-like system with a massive central object.

Two CUDA kernels are implemented:

- A simple brute-force version using only global memory  
- An optimized version using shared memory with a tiling approach  


### Simulation Preview

![Galaxy Simulation](galaxy_intro.gif)

---

## 2. CUDA Implementation Overview

The simulation computes gravitational forces between all pairs of particles.  
This results in a computational complexity of O(N²), which is common for direct N-body methods. The two different cuda implementations are the following:


### 2.1 Naive Kernel

In the naive approach, each CUDA thread updates one particle. This method is easy to implement but inefficient because the same particle data is repeatedly loaded from slow **global memory**.
So, memory bandwidth becomes the main bottleneck and performance drops quickly as the number of particles increases


### 2.2 Tiled Kernel Using Shared Memory

The optimized version divides the particles into small blocks (tiles). By reusing data stored in fast shared memory, the number of global memory accesses is greatly reduced.
This leads to a lower memory traffic, better GPU utilization, higher overall performance.

### 2.3 Barnes-Hut (Tree) (CPU Reference)

Fastest algorithm with O(N log N) complexity, but it is not implemented in CUDA due to high branching and irregular memory access patterns. It is implemented as a CPU reference for correctness and performance comparison.

### 2.4 Complexity Analysis

| Method | Memory Usage | Compute Complexity | Bandwidth Pressure |
|-------|-------------|------------------|-------------------|
| Naive | Global memory only | O(N²) | Very high |
| Tiled | Global + Shared memory | O(N²) | Reduced |
| Barnes-Hut (Tree) | Global memory + hierarchical tree (quadtree/octree), optional shared memory | O(N log N) |  |

---

## 3. Build and Run Instructions

### 3.1 Requirements
- NVIDIA CUDA Toolkit
- Python
- Python libraries listed in `requirements.txt`

Default simulation parameters:

- Particles: 1024
- Iterations: 10
- Steps per frame: 20
- Kernel: tiled

### 3.2 Compilation and Execution

```bash
nvidia-smi
pip install -r requirements.txt
make

# multiple iterations for 1024 objects, block size of 256
./nbody_sim -i 50000 -s 200 -m tiled 1> output.csv
or
./nbody_sim -i 50000 -s 200 -m naive 1> output.csv

python visualize.py -i output.csv
```

---

## 4. Results and Performance Comparison
###  4.1 Performance between tiled vs naive

The following plots show the performance comparison between the tiled and naive kernels across different block sizes. The tiled kernel demonstrates significantly better performance than the naive kernel in most cases, especially as the number of particles increases.

| Block Size 64 | Block Size 128 |
|---|---|
| ![Performance Block 64](benchmark_plots/performance_block_64.png) | ![Performance Block 128](benchmark_plots/performance_block_128.png) |

| Block Size 256 | Block Size 512 |
|---|---|
| ![Performance Block 256](benchmark_plots/performance_block_256.png) | ![Performance Block 512](benchmark_plots/performance_block_512.png) |

The tiled kernel consistently outperforms the naive kernel, showing particularly improvements as the particle count scales up. Thus, having multiple iterations make the output quicker.

This is due to the optimized use of shared memory, which reduces bandwidth pressure and improves arithmetic intensity.

We notice that the Barnes-Hut CPU algorithm is slower than both CUDA O(N²) implementations.

###  4.2 Execution Time

| Number of Particles | Naive (s) | Tiled (s) | Speedup |
|--------------------|----------|----------|---------|
| 512  | 0.0788 | 0.0640 | 1.23× |
| 1024 | 0.167  | 0.125 | 1.34× |
| 2048 | 0.342  | 0.246 | 1.39× |
| 4096 | 0.692  | 0.485 | 1.43× |
| 8192 | 1.378  | 0.962 | 1.43× |



###  4.3 Performance Observations

-   Tiled kernel minimizes redundant global memory reads
-   Speedup increases with particle count (and even iterations)

---

## 5. Conclusion

This project highlights the importance of memory optimization in CUDA programming (coelescing and shared memory). The optimized kernel reduces global memory traffic through data reuse in shared memory, leading to performance improvements while preserving output accuracy.

While the Barnes-Hut algorithm offers O(N log N) complexity, its branching and irregular memory access patterns make it less efficient than the tiled O(N²) approach in this implementation.

Overall, the project demonstrates how thoughtful use of CUDA’s memory hierarchy can lead to more scalable GPU applications.
