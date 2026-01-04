#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFULL

#define P1 73856093ULL
#define P2 19349663ULL
#define P3 83492791ULL

#define SHARED_TABLE_SIZE 1024

__device__ inline unsigned long long compute_voxel_hash(
    float px,
    float py,
    float pz,
    float voxel_size
) {
    int vx = (int)floorf(px / voxel_size);
    int vy = (int)floorf(py / voxel_size);
    int vz = (int)floorf(pz / voxel_size);

    unsigned long long hash_key = ((unsigned long long)(vx * P1))
        ^ ((unsigned long long)(vy * P2))
        ^ ((unsigned long long)(vz * P3));

    return hash_key;
}

__device__ inline void add_to_global(
    unsigned long long hash_key,
    float px, float py, float pz, int count,
    unsigned long long* table_keys,
    float* table_centroids,
    int* table_counts,
    int table_size
) {
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
            atomicAdd(&table_counts[table_idx], count);
            break;
        }
        table_idx = (table_idx + 1) % table_size;
    }
}

extern "C" __global__ void insert_points(
    const float* __restrict__ points,
    int num_points,
    float voxel_size,
    unsigned long long* __restrict__ table_keys,
    float* __restrict__ table_centroids,
    int * __restrict__ table_counts,
    int table_size
) {
    __shared__ unsigned long long s_keys[SHARED_TABLE_SIZE];
    __shared__ float s_centroids[SHARED_TABLE_SIZE * 3];
    __shared__ int s_counts[SHARED_TABLE_SIZE];

    for (int i = threadIdx.x; i < SHARED_TABLE_SIZE; i += blockDim.x) {
        s_keys[i] = EMPTY_KEY;
        s_centroids[i * 3 + 0] = 0.0f;
        s_centroids[i * 3 + 1] = 0.0f;
        s_centroids[i * 3 + 2] = 0.0f;
        s_counts[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float px = points[idx * 3 + 0];
        float py = points[idx * 3 + 1];
        float pz = points[idx * 3 + 2];
        
        unsigned long long hash_key = compute_voxel_hash(px, py, pz, voxel_size);
        
        int s_idx = hash_key % SHARED_TABLE_SIZE; 
        bool stored_in_shared = false;

        for (int i = 0; i < 32; ++i) { 
            unsigned long long old_s_key = atomicCAS(&s_keys[s_idx], EMPTY_KEY, hash_key);

            if (old_s_key == EMPTY_KEY || old_s_key == hash_key) {
                atomicAdd(&s_centroids[s_idx * 3 + 0], px);
                atomicAdd(&s_centroids[s_idx * 3 + 1], py);
                atomicAdd(&s_centroids[s_idx * 3 + 2], pz);
                atomicAdd(&s_counts[s_idx], 1);
                stored_in_shared = true;
                break;
            }
            s_idx = (s_idx + 1) % SHARED_TABLE_SIZE;
        }

        if (!stored_in_shared) {
            add_to_global(hash_key, px, py, pz, 1, 
                          table_keys, table_centroids, table_counts, table_size);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < SHARED_TABLE_SIZE; i += blockDim.x) {
        unsigned long long key = s_keys[i];
        
        if (key != EMPTY_KEY) {
            float sx = s_centroids[i * 3 + 0];
            float sy = s_centroids[i * 3 + 1];
            float sz = s_centroids[i * 3 + 2];
            int sc = s_counts[i];
            
            add_to_global(key, sx, sy, sz, sc,
                          table_keys, table_centroids, table_counts, table_size);
        }
    }
}

extern "C" __global__ void average_table(
    float* __restrict__ table_centroids,
    const int* __restrict__ table_counts,
    int table_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= table_size) return;

    int cnt = table_counts[idx];
    if (cnt > 1) {
        float inv_cnt = 1.0f / (float)cnt;
        table_centroids[idx * 3 + 0] *= inv_cnt;
        table_centroids[idx * 3 + 1] *= inv_cnt;
        table_centroids[idx * 3 + 2] *= inv_cnt;
    }
}

extern "C" __global__ void compact_voxels(
    const unsigned long long* __restrict__ table_keys,
    const float* __restrict__ table_centroids,
    const int* __restrict__ table_counts,
    int* __restrict__ table_remap,
    int table_size,
    float* __restrict__ out_points,
    int* __restrict__ out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= table_size) return;

    if (table_keys[idx] != EMPTY_KEY && table_counts[idx] > 0) {
        // int cnt = table_counts[idx];
        float sx = table_centroids[idx * 3 + 0];
        float sy = table_centroids[idx * 3 + 1];
        float sz = table_centroids[idx * 3 + 2];

        int write_idx = atomicAdd(out_count, 1);

        out_points[write_idx * 3 + 0] = sx;
        out_points[write_idx * 3 + 1] = sy;
        out_points[write_idx * 3 + 2] = sz;

        table_remap[idx] = write_idx;
    }
}

extern "C" __global__ void init_table(
    unsigned long long* __restrict__ table_keys,
    int* __restrict__ table_remap,
    int table_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= table_size) return;
    table_keys[idx] = EMPTY_KEY;
    table_remap[idx] = -1;
}