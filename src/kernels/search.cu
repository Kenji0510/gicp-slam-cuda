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

    float sx = source_pts[idx * 3 + 0];
    float sy = source_pts[idx * 3 + 1];
    float sz = source_pts[idx * 3 + 2];

    int best_idx = -1;
    float best_dist_sq = 1.0e30f;

    for (int j = 0; j < num_target; ++j) {
        float tx = target_pts[j * 3 + 0];
        float ty = target_pts[j * 3 + 1];
        float tz = target_pts[j * 3 + 2];

        float dx = sx - tx;
        float dy = sy - ty;
        float dz = sz - tz;

        float d2 = dx*dx + dy*dy + dz*dz;

        if (d2 < best_dist_sq) {
            best_dist_sq = d2;
            best_idx = j;
        }
    }

    out_indices[idx] = best_idx;
    out_dists_sq[idx] = best_dist_sq;
}