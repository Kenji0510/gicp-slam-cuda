# Compile the kernel
```bash
nvcc -ptx src/kernels/search.cu -o src/kernels/search.ptx \
    --gpu-architecture=compute_86 \
    --gpu-code=sm_86,sm_89,sm_90 \
    -std=c++17 \
    -O3 \
    --use_fast_math


nvcc -ptx src/kernels/compute_covariance.cu -o src/kernels/compute_covariance.ptx \
    --gpu-architecture=compute_86 \
    --gpu-code=sm_86,sm_89,sm_90 \
    -std=c++17 \
    -O3 \
    --use_fast_math

nvcc -ptx src/kernels/voxel.cu -o src/kernels/voxel.ptx \
    --gpu-architecture=compute_86 \
    --gpu-code=sm_86,sm_89,sm_90 \
    -std=c++17 \
    -O3 \
    --use_fast_math
```

# Logs
```bash
voxel_size = 0.5
Debug: v_source points: 1105, v_target points: 1105
KNN search took: 88.857µs
Calculating covariance took: 714.342µs
Debug: v_source points: 1118, v_target points: 1118
KNN search took: 90.55µs
Calculating covariance took: 713.41µs
Debug: v_source points: 1149, v_target points: 1149
KNN search took: 93.707µs
Calculating covariance took: 736.714µs
Debug: v_source points: 1151, v_target points: 1151

voxel_size = 0.1
Debug: v_source points: 5556, v_target points: 5556
KNN search took: 392.497µs
Calculating covariance took: 2.264224ms
Debug: v_source points: 5506, v_target points: 5506
KNN search took: 388.681µs
Calculating covariance took: 2.293126ms
Debug: v_source points: 5562, v_target points: 5562
KNN search took: 392.889µs
Calculating covariance took: 2.286606ms
Debug: v_source points: 5624, v_target points: 5624
```

Trajectory Record:
  Min Distance: 0.0000 m
  Max Distance: 0.3159 m
  Min Degree   : 0.0000 deg
  Max Degree   : 6.4101 deg