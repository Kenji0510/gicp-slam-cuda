# Compile the kernel
```bash
nvcc -ptx src/kernels/search.cu -o src/kernels/search.ptx \
    --gpu-architecture=compute_86 \
    --gpu-code=sm_86,sm_89,sm_90 \
    -std=c++17 \
    -O3 \
    --use_fast_math
```

# Logs
```bash
Debug: source points: 19937, target points: 19937
KNN search took: 1.532227ms
Debug: source points: 20024, target points: 20024
KNN search took: 1.538529ms
Debug: source points: 19922, target points: 19922


Debug: v_source points: 5691, v_target points: 5691
KNN search took: 436.689µs
Debug: v_source points: 5670, v_target points: 5670
KNN search took: 436.308µs
Debug: v_source points: 5654, v_target points: 5654
KNN search took: 434.424µs
Debug: v_source points: 5695, v_target points: 5695

Debug: v_source points: 1182, v_target points: 1182
KNN search took: 94.367µs
Debug: v_source points: 1182, v_target points: 1182
KNN search took: 94.347µs
Debug: v_source points: 1166, v_target points: 1166
KNN search took: 93.305µs
Debug: v_source points: 1180, v_target points: 1180
```