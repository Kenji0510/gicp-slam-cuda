#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFULL

#define P1 73856093ULL
#define P2 19349663ULL
#define P3 83492791ULL

extern "C" __global__ void insert_points(
    const float* __restrict__ points,
    int num_points,
    float voxel_size,
    unsigned long long* __restrict__ table_keys,
    float* __restrict__ table_centroids,
    int * __restrict__ table_counts,
    int table_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float px = points[idx * 3 + 0];;
    float py = points[idx * 3 + 1];;
    float pz = points[idx * 3 + 2];

    int vx = (int)floorf(px / voxel_size);
    int vy = (int)floorf(py / voxel_size);
    int vz = (int)floorf(pz / voxel_size);

    unsigned long long hash_key = ((unsigned long long)(vx * P1))
        ^ ((unsigned long long)(vy * P2))
        ^ ((unsigned long long)(vz * P3));

    int table_idx = hash_key % table_size;

    for (int i = 0; i < 1000; ++i) {
        unsigned long long old_key = atomicCAS(
            &table_keys[table_idx],
            EMPTY_KEY,
            hash_key
        );

        if (old_key == EMPTY_KEY || old_key == hash_key) {
            atomicAdd(&table_centroids[table_idx * 3 + 0], px);
            atomicAdd(&table_centroids[table_idx * 3 + 1], py);
            atomicAdd(&table_centroids[table_idx * 3 + 2], pz);
            atomicAdd(&table_counts[table_idx], 1);
            break;
        }

        table_idx = (table_idx + 1) % table_size;
    }
}

extern "C" __global__ void compact_voxels(
    const unsigned long long* __restrict__ table_keys,
    const float* __restrict__ table_centroids,
    const int* __restrict__ table_counts,
    int table_size,
    float* __restrict__ out_points,
    int* __restrict__ out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= table_size) return;

    if (table_keys[idx] != EMPTY_KEY && table_counts[idx] > 0) {
        int cnt = table_counts[idx];
        float sx = table_centroids[idx * 3 + 0];
        float sy = table_centroids[idx * 3 + 1];
        float sz = table_centroids[idx * 3 + 2];

        int write_idx = atomicAdd(out_count, 1);

        out_points[write_idx * 3 + 0] = sx / cnt;;
        out_points[write_idx * 3 + 1] = sy / cnt;
        out_points[write_idx * 3 + 2] = sz / cnt;
    }
}

extern "C" __global__ void init_table(
    unsigned long long* __restrict__ table_keys,
    int table_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= table_size) return;
    table_keys[idx] = EMPTY_KEY;
}