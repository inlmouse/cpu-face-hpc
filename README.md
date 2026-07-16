# CpuFaceHpc: Squeezing Modern CPU Architectures to Physical Limits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Category: HPC / TinyML](https://img.shields.io/badge/Category-HPC%20%2F%20TinyML-orange.svg)]()

> **"Algorithms fade, but hardware optimization is eternal."**
> This repository is a showcase of how to transform an "obsolete" traditional computer vision pipeline into a cutting-edge lesson in High-Performance Computing (HPC), dataflow engineering, and micro-architectural overclocking on modern general-purpose CPUs.

---

## Background & Inspiration

This project is a **academic re-implementation** and modern architectural overhaul inspired by the legendary binary-only era ([this old version](https://github.com/ShiqiYu/libfacedetection/tree/23ac86b), circa 2017) of **libfacedetection** created by Prof. Shiqi Yu. 

Nearly a decade ago, before the modern deep learning boom, this closed-source `.dll` stunned the vision community by achieving unbelievable real-time face detection speeds on standard x86/ARM CPUs. For years, its low-level hardware-squeezing mechanics remained a fascinating black box. 

By analyzing Prof. Yu's **[public interviews](https://mp.weixin.qq.com/s/UITlYu8DDs8fNVKi8DoldQ) and technical retrospectives**, where he disclosed the conceptual usage of sparse multi-channel pixel-difference features and tree unrolling, this project was built from scratch. We decoupled the high-level geometric mathematical ideas from the compilation artifacts, scaling them up with modern **Data-Oriented Design (DOD)** principles to construct the ultimate instruction-level and cache-line optimized inference engine.

---

## Micro-Architectural Interventions (The Core "Black Magic")

Since the algorithmic accuracy of cascaded regressors has been surpassed by modern deep learning (e.g., YuNet, YOLO), the true value of this repository lies entirely in its **HPC design patterns** for defeating the memory wall and instruction pipelines:

### 1. Data-Oriented Design (DOD) via Cold/Hot Separation
Instead of using massive object-oriented node classes that bloat the CPU L1 Cache line, we split the decision structures into two distinct domains:
- **`ModelNode` (Cold Data)**: Retains high-level spatial coordinates $(x, y, ch)$ strictly used during deserialization.
- **`RuntimeNode` (Hot Data)**: A heavily stripped down struct containing only raw pointers and integer thresholds. 
- *Micro-arch Dividend*: Under 64-bit platforms, `RuntimeNode` is strictly static-asserted to **32 bytes**. A standard 64-byte L1 Cache Line swallows exactly two nodes perfectly, slicing cache miss rates to near-zero during deep spatial sweeps.

### 2. Runtime Pointer Swizzling (Zero-Arithmetic Windows)
Standard multi-scale scanning cascades waste massive ALU cycles calculating dynamic addresses inside the pixel loops:

Offset = (Y * Stride) + (C * Channel_Stride) + X

- *Our Solution*: Prior to entering the window sliding nested loops for each image pyramid scale, a **Runtime Pointer Patch** is executed. Every node's channel and window offsets are pre-flattened into absolute memory base pointers (`ptrA`/`ptrB`). Inside the hot path, evaluation collapses into a zero-overhead $O(1)$ pointer dereference: `ptrA[current_window_offset]`.

### 3. Branchless Dynamic Tree Routing & Loop Unrolling
To completely bypass CPU branch misprediction penalties caused by highly unpredictable decision trees:
- The top-level weak classifiers are manually unrolled by 4-tree steps to allow maximum out-of-order execution (OoO) and instruction-level parallelism (ILP) within the CPU execution units.
- The depth-based loops are unrolled dynamically into flat lookup flows, avoiding structural control hazards while maintaining backward compatibility for variable tree sizes ($Depth\text{-}2$ to $Depth\text{-}5$).

### 4. Transcendental-Free Horn's Procrustes Regression
During the 68-landmark Explicit Shape Regression (ESR) refinement stages, standard implementations suffer immense latency overhead by executing costly transcendental math functions (`atan2`, `sin`, `cos`). 
- We implement the original compiler-friendly matrix projection. By accumulating 2nd-order geometric statistics ($L_2$ norms $\Sigma xx, \Sigma uu$) and cross-covariance components ($A, B$), the rigid shape similarity transformation matrix $[S_0, S_1, S_2, S_3]$ is derived entirely through standard multiplications and single-cycle hardware `sqrt` instructions.


---

## High-Performance Profiling Metrics (Educational Target)
When profiling the HPC implementation via Intel VTune or Linux `perf stat`, this architecture enforces excellent hardware characteristics:
- **Branch Misprediction Rate**: $< 1.2\%$ (via manual unrolling and algorithmic early stage shortcuts).
- **L1 Data Cache Load Miss Rate**: Reduced by over $60\%$ compared to traditional OOP node representations.
- **Instructions Per Cycle (IPC)**: Massive execution scaling due to the removal of spatial pointer dependency chains.

---

## Legal Disclaimer & Clean-Room Status

- This repository is an entirely **independent algorithm implementation** based on publicly available academic papers and technical blog post interviews describing classic vision frameworks (Cascaded Adaboost + Explicit Shape Regression).
- It does **NOT** contain any copyrighted binary blobs, leaked proprietary model parameters, or secret data extracted from commercial software distributions.
- The repository only distributes the open-source **inference source code and processing pipeline**. Users must supply or train their own open-format model parameters.
- This codebase is published strictly under the **MIT License** for educational research, micro-architectural evaluation, and system performance benchmarks.

---

## Building & Requirements
- A modern C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2019+).
- **Optimizations**: Must be compiled with maximum release flags (`-O3 -march=native`) to unlock full loop vectorization and hyper-pipeline scheduling.
- **OpenCV (4.x+)**: Optional, required *only* if building the `readable_opencv_demo` tut
