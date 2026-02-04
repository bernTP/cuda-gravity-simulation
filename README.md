# CUDA N-Body Galaxy Simulation

##  1. Project Overview

## 🔍 Technical Summary

This project implements a GPU-accelerated N-body gravitational simulation using CUDA.  

Two kernels are developed:

- A baseline brute-force implementation using global memory
- An optimized tiled implementation using shared memory

The optimized kernel reduces global memory traffic through data reuse in shared memory, leading to significant performance improvements while preserving physical accuracy.

Key focus areas:

- CUDA memory hierarchy optimization
- Parallel force computation
- Performance scalability analysis
- 3D visualization of particle dynamics

**Simulation Preview**

![Galaxy Simulation](galaxy_fixed.gif)

---

## 2. CUDA Implementation Techniques

The simulation computes gravitational forces using a direct all-pairs interaction. This leads to an O(N²) computational complexity. Two CUDA strategies are implemented.

---

### 2.1 Naive Global Memory Kernel

Each CUDA thread is responsible for updating one particle.

#### Workflow:

- Each thread loads all particles directly from **global memory**
- Computes pairwise forces sequentially
- Updates velocity and position

#### Key properties:

- Each particle is read N times by different threads
- Heavy pressure on global memory bandwidth
- No shared memory usage

#### Performance impact:

- Poor scalability
- Quickly becomes bandwidth-bound

---

### 2.2 Tiled Kernel with Shared Memory Optimization

The tiled version partitions the particle set into blocks of size `block_size`.

#### Workflow:
Each block:

1. Loads a tile of particles from global memory into **shared memory**
2. Synchronizes threads using `__syncthreads()`
3. Computes interactions between the local particle and the tile
4. Iterates over all tiles

#### Key properties:

- Drastically reduced global memory transactions
- High data reuse
- Improved cache efficiency

#### Performance impact:
- Reduced global memory bandwidth pressure
- Improved GPU occupancy and throughput
---

### 🔍 Complexity Analysis

| Method | Memory Access Pattern | Compute Complexity | Bandwidth Usage |
|-------|---------------------|------------------|----------------|
| Naive | Global memory only | O(N²) | Very high |
| Tiled | Global + Shared memory | O(N²) | Reduced |

---

## 3. Build & Run Instructions

###  Requirements

- NVIDIA CUDA Toolkit
- Python
- Python dependencies
- Default parameters:
	-   Particles: 1024
	-   Iterations: 10
	-   Steps per frame: 20
	-   Kernel: tiled

```bash
nvidia-smi
pip install -r requirements.txt
make

./nbody_sim -n 1024 -s 20 -i 10 -m tiled > output.csv 
or
./nbody_sim -n 1024 -s 20 -i 10 -m naive > output.csv 

python visualize.py -i output.csv
```

##  4. Results and Performance Comparison
###  4.1 Visual Output (with Default parameters)
| Naive Kernel | Tiled Kernel |
|--------------------|----------|
| ![Galaxy Simulation](galaxy_naive.gif)  | ![Galaxy Simulation](galaxy_tiled.gif) |


Both kernels produce equivalent physical trajectories.  
The major difference lies in execution speed.

----------

###  4.2 Execution Time

| Number of Particles | Naive (s) | Tiled (s) | Speedup |
|--------------------|----------|----------|---------|
| 512  | 0.0788 | 0.0640 | 1.23× |
| 1024 | 0.167  | 0.125 | 1.34× |
| 2048 | 0.342  | 0.246 | 1.39× |
| 4096 | 0.692  | 0.485 | 1.43× |
| 8192 | 1.378  | 0.962 | 1.43× |

----------

###  4.3 Performance Observations

-   Naive kernel becomes memory-bound very quickly
-   Tiled kernel minimizes redundant global memory reads
-   Speedup increases with particle count
-   Shared memory significantly improves arithmetic intensity
----------

## 5. Conclusion

This project demonstrates the critical role of memory optimization in CUDA applications:

-   Direct N² implementations are simple but inefficient
-   Shared memory tiling drastically improves performance
-   Proper use of CUDA memory hierarchy is essential for scalable GPU computing
    
The tiled kernel achieves substantial acceleration while preserving numerical correctness.

## 6. Limitations and Future Work

While the tiled shared memory optimization significantly improves performance, the simulation still relies on a direct O(N²) force computation.

Potential improvements include:

- Barnes-Hut algorithm for O(N log N) complexity
- Multi-GPU parallelization
- Further memory coalescing optimizations


These approaches would enable simulations with much larger particle counts.