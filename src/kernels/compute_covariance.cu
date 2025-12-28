#define K 20

extern "C" __global__ void compute_covariance(
    const float* __restrict__ points,
    int num_points,
    float* __restrict__ out_covariances
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float px = points[idx * 3 + 0];
    float py = points[idx * 3 + 1];
    float pz = points[idx * 3 + 2];

    int neighbor_indices[K];
    float neighbor_dists[K];

    for (int i = 0; i < K; ++i) {
        neighbor_dists[i] = 1.030f;
        neighbor_indices[i] = -1;
    }

    for (int j = 0; j < num_points; ++j) {
        if (idx == j) continue;

        float tx = points[j * 3 + 0];
        float ty = points[j * 3 + 1];
        float tz = points[j * 3 + 2];

        float dx = px - tx;
        float dy = py - ty;
        float dz = pz - tz;
        float d2 = dx*dx + dy*dy + dz*dz;

        if (d2 < neighbor_dists[K - 1]) {
            int insert_pos = K - 1;
            while (insert_pos > 0 && d2 < neighbor_dists[insert_pos - 1]) {
                neighbor_dists[insert_pos] = neighbor_dists[insert_pos - 1];
                neighbor_indices[insert_pos] = neighbor_indices[insert_pos - 1];
                insert_pos--;
            }
            neighbor_dists[insert_pos] = d2;
            neighbor_indices[insert_pos] = j;
        }
    }

    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < K; ++i) {
        int n_idx = neighbor_indices[i];
        if (n_idx == -1) break;

        sum_x += points[n_idx * 3 + 0];
        sum_y += points[n_idx * 3 + 1];
        sum_z += points[n_idx * 3 + 2];
        valid_count++;
    }

    if (valid_count < 3) {
        out_covariances[idx * 9 + 0] = 0.0f; out_covariances[idx * 9 + 1] = 0.0f; out_covariances[idx * 9 + 2] = 0.0f;
        out_covariances[idx * 9 + 3] = 0.0f; out_covariances[idx * 9 + 4] = 0.0f; out_covariances[idx * 9 + 5] = 0.0f;
        out_covariances[idx * 9 + 6] = 0.0f; out_covariances[idx * 9 + 7] = 0.0f; out_covariances[idx * 9 + 8] = 0.0f;
        return;
    }

    float mean_x = sum_x / valid_count;
    float mean_y = sum_y / valid_count;
    float mean_z = sum_z / valid_count;

    float c_xx = 0.0f, c_xy = 0.0f, c_xz = 0.0f;
    float c_yy = 0.0f, c_yz = 0.0f, c_zz = 0.0f;

    for (int i = 0; i < valid_count; ++i) {
        int n_idx = neighbor_indices[i];
        float dx = points[n_idx * 3 + 0] - mean_x;
        float dy = points[n_idx * 3 + 1] - mean_y;
        float dz = points[n_idx * 3 + 2] - mean_z;

        c_xx += dx * dx;
        c_xy += dx * dy;
        c_xz += dx * dz;
        c_yy += dy * dy;
        c_yz += dy * dz;
        c_zz += dz * dz;
    }

    float inv_k = 1.0f / valid_count;

    out_covariances[idx * 9 + 0] = c_xx * inv_k;
    out_covariances[idx * 9 + 1] = c_xy * inv_k;
    out_covariances[idx * 9 + 2] = c_xz * inv_k;
    out_covariances[idx * 9 + 3] = c_xy * inv_k;
    out_covariances[idx * 9 + 4] = c_yy * inv_k;
    out_covariances[idx * 9 + 5] = c_yz * inv_k;
    out_covariances[idx * 9 + 6] = c_xz * inv_k;
    out_covariances[idx * 9 + 7] = c_yz * inv_k;
    out_covariances[idx * 9 + 8] = c_zz * inv_k;
}