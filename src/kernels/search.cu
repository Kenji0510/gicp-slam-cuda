// #define BLOCK_DIM 256
#define BLOCK_DIM 128

extern "C" __global__ void find_nearest_neighbor(
    const float* __restrict__ source_pts,
    const float* __restrict__ target_pts,
    int num_source,
    int num_target,
    int* __restrict__ out_indices,
    float* __restrict__ out_dists_sq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_source) return;

    float3 p_s;
    if (idx < num_source) {
        p_s.x = source_pts[idx * 3 + 0];
        p_s.y = source_pts[idx * 3 + 1];
        p_s.z = source_pts[idx * 3 + 2];
    }

    int best_idx = -1;
    float best_dist_sq = 1.0e30f;

    __shared__ float3 s_target_pts[BLOCK_DIM];

    for (int tile_start = 0; tile_start < num_target; tile_start += BLOCK_DIM) {
        int t_idx = tile_start + threadIdx.x;

        if (t_idx < num_target) {
            s_target_pts[threadIdx.x].x = target_pts[t_idx * 3 + 0];
            s_target_pts[threadIdx.x].y = target_pts[t_idx * 3 + 1];
            s_target_pts[threadIdx.x].z = target_pts[t_idx * 3 + 2];
        }

        __syncthreads();

        if (idx < num_source) {
            int limit = min(BLOCK_DIM, num_target - tile_start);

            #pragma unroll
            for (int k = 0; k < limit; ++k) {
                float3 p_t = s_target_pts[k];

                float dx = p_s.x - p_t.x;
                float dy = p_s.y - p_t.y;
                float dz = p_s.z - p_t.z;

                float d2 = dx*dx + dy*dy + dz*dz;

                if (d2 < best_dist_sq) {
                    best_dist_sq = d2;
                    best_idx = tile_start + k;
                }
            }
        }

        __syncthreads();
    }

    if (idx < num_source) {
        out_dists_sq[idx] = best_dist_sq;
        out_indices[idx] = best_idx;
    }
}