# CUDA N-Body Galaxy Simulation

##  1. Project Overview

This project implements a three-dimensional **N-body gravitational simulation** using CUDA to model the dynamic evolution of a galaxy-like system.  
Each particle represents a star interacting gravitationally with all others, with a massive central body acting as a black hole.

The main goals are:
- Implement a baseline brute-force CUDA kernel (*naive*)
- Optimize force computation using shared memory (*tiling*)
- Analyze performance improvements and scalability
- Visualize the simulation in 3D using Python

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
- Better GPU occupancy

---

### 🔍 Complexity Analysis

| Method | Memory Access Pattern | Compute Complexity | Bandwidth Usage |
|-------|---------------------|------------------|----------------|
| Naive | Global memory only | O(N²) | Very high |
| Tiled | Global + Shared memory | O(N²/tile) | Reduced |

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
pip install -r requirements.txt
make

./nbody_sim -n 1024 -s 20 -i 10 -m tiled > output.csv 
or
./nbody_sim -n 1024 -s 20 -i 10 -m naive > output.csv 

python visualize.py -i output.csv
```

##  4. Results and Performance Comparison
###  4.1 Visual Output
#### 4.1.1 Naive Kernel (with Default parameters)
![Galaxy Simulation](galaxy_naive.gif)
#### 4.1.2 Tiled Kernel (with Default parameters)
![Galaxy Simulation](galaxy_tiled.gif)


Both kernels produce equivalent physical trajectories.  
The major difference lies in execution speed.

----------

###  4.2 Execution Time

| Number of Particles | Naive (s) |  Tiled (s)|
|-------|---------------------|------------------|
| 512  | 0.0788 | 0.0640 |
| 1024 | 0.167  | 0.125 |
| 2048 | 0.342  | 0.246 |
| 4096 | 0.692  | 0.485 |
| 8192 | 1.378  | 0.962 |

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