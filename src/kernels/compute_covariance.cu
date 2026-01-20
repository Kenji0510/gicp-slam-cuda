#define K 12

__device__ void eigen_decomposition(float A[3][3], float evecs[3][3], float evals[3]) {
    // 単位行列で初期化
    evecs[0][0] = 1.0f; evecs[0][1] = 0.0f; evecs[0][2] = 0.0f;
    evecs[1][0] = 0.0f; evecs[1][1] = 1.0f; evecs[1][2] = 0.0f;
    evecs[2][0] = 0.0f; evecs[2][1] = 0.0f; evecs[2][2] = 1.0f;

    // Jacobi Iteration (通常3~4回のスイープで収束する)
    int max_iter = 15;
    for (int iter = 0; iter < max_iter; ++iter) {
        float max_val = 0.0f;
        int p = 0, q = 1;

        // 非対角要素の最大値を探す
        float a01 = fabsf(A[0][1]);
        float a02 = fabsf(A[0][2]);
        float a12 = fabsf(A[1][2]);

        if (a01 >= a02 && a01 >= a12) { p = 0; q = 1; max_val = a01; }
        else if (a02 >= a01 && a02 >= a12) { p = 0; q = 2; max_val = a02; }
        else { p = 1; q = 2; max_val = a12; }

        if (max_val < 1e-6f) break; // 十分小さいなら終了

        float app = A[p][p];
        float aqq = A[q][q];
        float apq = A[p][q];

        float phi = 0.5f * atan2f(2.0f * apq, aqq - app);
        float c = cosf(phi);
        float s = sinf(phi);

        // A' = J^T * A * J の更新
        A[p][p] = c*c*app - 2.0f*s*c*apq + s*s*aqq;
        A[q][q] = s*s*app + 2.0f*s*c*apq + c*c*aqq;
        A[p][q] = 0.0f;
        A[q][p] = 0.0f;

        float arp, arq;
        for (int r = 0; r < 3; ++r) {
            if (r != p && r != q) {
                arp = A[r][p];
                arq = A[r][q];
                A[r][p] = c*arp - s*arq;
                A[p][r] = A[r][p];
                A[r][q] = s*arp + c*arq;
                A[q][r] = A[r][q];
            }
        }

        // 固有ベクトルの更新: V' = V * J
        for (int r = 0; r < 3; ++r) {
            float vrp = evecs[r][p];
            float vrq = evecs[r][q];
            evecs[r][p] = c*vrp - s*vrq;
            evecs[r][q] = s*vrp + c*vrq;
        }
    }

    evals[0] = A[0][0];
    evals[1] = A[1][1];
    evals[2] = A[2][2];
}

__device__ void recompute_max_k(
    float* dists,
    float* dmax, int* imax)
{
    float dm = dists[0];
    int im = 0;
    for (int t = 1; t < K; ++t) {
        float v = dists[t];
        if (v > dm) { dm = v; im = t; }
    }
    *dmax = dm;
    *imax = im;
}

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
        // neighbor_dists[i] = 1.030f;
        neighbor_dists[i] = 100.0f;
        neighbor_indices[i] = -1;
    }

    float dmax = neighbor_dists[0];
    int imax = 0;

    recompute_max_k(neighbor_dists, &dmax, &imax);

    for (int j = 0; j < num_points; ++j) {
        if (idx == j) continue;

        float tx = points[j * 3 + 0];
        float ty = points[j * 3 + 1];
        float tz = points[j * 3 + 2];

        float dx = px - tx;
        float dy = py - ty;
        float dz = pz - tz;
        float d2 = dx*dx + dy*dy + dz*dz;

        if (d2 < dmax) {
            neighbor_dists[imax] = d2;
            neighbor_indices[imax] = j;

            recompute_max_k(neighbor_dists, &dmax, &imax);
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
        out_covariances[idx * 9 + 0] = 1.0f; out_covariances[idx * 9 + 1] = 0.0f; out_covariances[idx * 9 + 2] = 0.0f;
        out_covariances[idx * 9 + 3] = 0.0f; out_covariances[idx * 9 + 4] = 1.0f; out_covariances[idx * 9 + 5] = 0.0f;
        out_covariances[idx * 9 + 6] = 0.0f; out_covariances[idx * 9 + 7] = 0.0f; out_covariances[idx * 9 + 8] = 1.0f;
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

    float mat[3][3];
    mat[0][0] = c_xx * inv_k; mat[0][1] = c_xy * inv_k; mat[0][2] = c_xz * inv_k;
    mat[1][0] = c_xy * inv_k; mat[1][1] = c_yy * inv_k; mat[1][2] = c_yz * inv_k;
    mat[2][0] = c_xz * inv_k; mat[2][1] = c_yz * inv_k; mat[2][2] = c_zz * inv_k;

    // Eigen Decomposition
    float evecs[3][3];
    float evals[3];
    eigen_decomposition(mat, evecs, evals);

    int min_idx = 0;
    if (evals[1] < evals[min_idx]) min_idx = 1;
    if (evals[2] < evals[min_idx]) min_idx = 2;

    float reg_evals[3];
    reg_evals[0] = 1.0f; reg_evals[1] = 1.0f; reg_evals[2] = 1.0f;
    reg_evals[min_idx] = 1e-3f;

    // Reconstruct C = V * diag(reg_evals) * V^T
    // C_new = Sum( lambda_i * v_i * v_i^T )
    
    float r_cov[3][3] = {0};

    for (int k = 0; k < 3; ++k) {
        float lambda = reg_evals[k];
        float vx = evecs[0][k];
        float vy = evecs[1][k];
        float vz = evecs[2][k];

        r_cov[0][0] += lambda * vx * vx;
        r_cov[0][1] += lambda * vx * vy;
        r_cov[0][2] += lambda * vx * vz;
        
        r_cov[1][1] += lambda * vy * vy;
        r_cov[1][2] += lambda * vy * vz;
        
        r_cov[2][2] += lambda * vz * vz;
    }
    // Fill Symmetric parts
    r_cov[1][0] = r_cov[0][1];
    r_cov[2][0] = r_cov[0][2];
    r_cov[2][1] = r_cov[1][2];

    // Write back to global memory
    out_covariances[idx * 9 + 0] = r_cov[0][0];
    out_covariances[idx * 9 + 1] = r_cov[0][1];
    out_covariances[idx * 9 + 2] = r_cov[0][2];

    out_covariances[idx * 9 + 3] = r_cov[1][0];
    out_covariances[idx * 9 + 4] = r_cov[1][1];
    out_covariances[idx * 9 + 5] = r_cov[1][2];

    out_covariances[idx * 9 + 6] = r_cov[2][0];
    out_covariances[idx * 9 + 7] = r_cov[2][1];
    out_covariances[idx * 9 + 8] = r_cov[2][2];
}