// #define BLOCK_DIM 256
#define BLOCK_DIM 128

#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFULL

#define P1 73856093ULL
#define P2 19349663ULL
#define P3 83492791ULL


__device__ inline unsigned long long compute_hash(
    int vx, int vy, int vz
) {
    return ((unsigned long long)(vx * P1))
        ^ ((unsigned long long)(vy * P2))
        ^ ((unsigned long long)(vz * P3));
}

__device__ inline int lookup_table(
    unsigned long long key,
    const unsigned long long* __restrict__ table_keys,
    int table_size
) {
    int idx = key % table_size;
    for (int i = 0; i < 50; ++i) {
        unsigned long long k = table_keys[idx];
        if (k == key) return idx;
        if (k == EMPTY_KEY) return -1;
        idx = (idx + 1) % table_size;
    }
    return -1;
}

extern "C" __global__ void find_nearest_neighbor(
    const float* __restrict__ source_pts,
    int num_source,
    float voxel_size,
    const unsigned long long* __restrict__ target_keys,
    const float* __restrict__ target_centroids,
    const int* __restrict__ target_counts,
    const int* __restrict__ table_remap,
    int table_size,
    int* __restrict__ out_indices,
    float* __restrict__ out_dists_sq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_source) return;

    float px = source_pts[idx * 3 + 0];
    float py = source_pts[idx * 3 + 1];
    float pz = source_pts[idx * 3 + 2];

    int vx = (int)floorf(px / voxel_size);
    int vy = (int)floorf(py / voxel_size);
    int vz = (int)floorf(pz / voxel_size);

    float best_dist_sq = 1.0e30f;
    int best_table_idx = -1;

    int search_range = 1;

    for (int dz = -search_range; dz <= search_range; ++dz) {
        for (int dy = -search_range; dy <= search_range; ++dy) {
            for (int dx = -search_range; dx <= search_range; ++dx) {
                unsigned long long key = compute_hash(vx + dx, vy + dy, vz + dz);
                int t_idx = lookup_table(
                    key,
                    target_keys,
                    table_size
                );

                if (t_idx != -1 && target_counts[t_idx] > 0) {
                    float tx = target_centroids[t_idx * 3 + 0];
                    float ty = target_centroids[t_idx * 3 + 1];
                    float tz = target_centroids[t_idx * 3 + 2];

                    // int cnt = target_counts[t_idx];
                    // tx /= cnt;
                    // ty /= cnt;
                    // tz /= cnt;

                    float diff_x = px - tx;
                    float diff_y = py - ty;
                    float diff_z = pz - tz;
                    float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                    if (dist_sq < best_dist_sq) {
                        best_dist_sq = dist_sq;
                        best_table_idx = t_idx;
                    }
                }
            }
        }
    }

    out_dists_sq[idx] = best_dist_sq;
    if (best_table_idx != -1) {
        out_indices[idx] = table_remap[best_table_idx];
    } else {
        out_indices[idx] = -1;
    }
}