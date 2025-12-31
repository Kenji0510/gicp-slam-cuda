#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ __forceinline__
bool invert3x3Sym_safe(const float* src, float* dst) {
    // src, dst: row-major 3x3
    float det =
        src[0] * (src[4] * src[8] - src[5] * src[7]) -
        src[1] * (src[3] * src[8] - src[5] * src[6]) +
        src[2] * (src[3] * src[7] - src[4] * src[6]);

    if (fabsf(det) < 1e-12f) return false;
    float invDet = 1.0f / det;

    dst[0] = (src[4] * src[8] - src[5] * src[7]) * invDet;
    dst[1] = (src[2] * src[7] - src[1] * src[8]) * invDet;
    dst[2] = (src[1] * src[5] - src[2] * src[4]) * invDet;

    dst[3] = dst[1];
    dst[4] = (src[0] * src[8] - src[2] * src[6]) * invDet;
    dst[5] = (src[2] * src[3] - src[0] * src[5]) * invDet;

    dst[6] = dst[2];
    dst[7] = dst[5];
    dst[8] = (src[0] * src[4] - src[1] * src[3]) * invDet;

    return true;
}

__device__ __forceinline__
void mul3(const float* A, const float* v, float* out) {
    // A: row-major 3x3, v: 3x1
    out[0] = A[0]*v[0] + A[1]*v[1] + A[2]*v[2];
    out[1] = A[3]*v[0] + A[4]*v[1] + A[5]*v[2];
    out[2] = A[6]*v[0] + A[7]*v[1] + A[8]*v[2];
}

extern "C" __global__
void compute_gicp_linear_system(
    const float* __restrict__ d_src_pts,   // (num_source x 3)
    const float* __restrict__ d_src_covs,  // (num_source x 9) row-major
    const float* __restrict__ d_tgt_pts,   // (num_target x 3)
    const float* __restrict__ d_tgt_covs,  // (num_target x 9) row-major
    const int*   __restrict__ d_indices,   // (num_source)
    const float* __restrict__ d_dists_sq,  // (num_source)
    int   num_source,
    int   num_target,
    float max_dist_sq,
    float* __restrict__ d_H,               // (36) row-major 6x6 (global accumulated)
    float* __restrict__ d_b                // (6)  (global accumulated)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // per-thread local accumulators
    float local_H[36];
    float local_b[6];
#pragma unroll
    for (int i=0;i<36;i++) local_H[i] = 0.0f;
#pragma unroll
    for (int i=0;i<6;i++)  local_b[i] = 0.0f;

    if (idx < num_source) {
        int target_idx = d_indices[idx];
        float dist_sq  = d_dists_sq[idx];

        if (target_idx >= 0 && target_idx < num_target && dist_sq <= max_dist_sq) {
            // ps, pt
            float ps[3], pt[3];
            ps[0] = d_src_pts[idx * 3 + 0];
            ps[1] = d_src_pts[idx * 3 + 1];
            ps[2] = d_src_pts[idx * 3 + 2];

            pt[0] = d_tgt_pts[target_idx * 3 + 0];
            pt[1] = d_tgt_pts[target_idx * 3 + 1];
            pt[2] = d_tgt_pts[target_idx * 3 + 2];

            // C_sum = C_t + C_s (Rustと一致：Rで回してない前提)
            float C_sum[9];
#pragma unroll
            for (int k=0;k<9;k++) {
                C_sum[k] = d_tgt_covs[target_idx * 9 + k] + d_src_covs[idx * 9 + k];
            }

            // optional: 正則化（必要なら有効化）
            // C_sum[0] += 1e-3f; C_sum[4] += 1e-3f; C_sum[8] += 1e-3f;

            // Omega = inv(C_sum)
            float Omega[9];
            bool ok = invert3x3Sym_safe(C_sum, Omega);
            if (ok) {
                // error = p_t - p_s
                float err[3] = {
                    pt[0] - ps[0],
                    pt[1] - ps[1],
                    pt[2] - ps[2]
                };

                // We = Omega * err
                float We[3];
                mul3(Omega, err, We);

                float x = ps[0], y = ps[1], z = ps[2];

                // b_rot = p × We  (Rustと一致)
                local_b[0] = y*We[2] - z*We[1];
                local_b[1] = z*We[0] - x*We[2];
                local_b[2] = x*We[1] - y*We[0];
                // b_trans = We
                local_b[3] = We[0];
                local_b[4] = We[1];
                local_b[5] = We[2];

                // s_col vectors (Rustと一致)
                float s0[3] = {0.0f, -z,  y};
                float s1[3] = { z,  0.0f, -x};
                float s2[3] = {-y,   x,  0.0f};

                // ws = Omega * s_col
                float ws0[3], ws1[3], ws2[3];
                mul3(Omega, s0, ws0);
                mul3(Omega, s1, ws1);
                mul3(Omega, s2, ws2);

                // H_rr columns = p × ws{0,1,2} (Rustと一致：マイナス不要)
                float hcol0[3], hcol1[3], hcol2[3];
                hcol0[0] = y*ws0[2] - z*ws0[1];
                hcol0[1] = z*ws0[0] - x*ws0[2];
                hcol0[2] = x*ws0[1] - y*ws0[0];

                hcol1[0] = y*ws1[2] - z*ws1[1];
                hcol1[1] = z*ws1[0] - x*ws1[2];
                hcol1[2] = x*ws1[1] - y*ws1[0];

                hcol2[0] = y*ws2[2] - z*ws2[1];
                hcol2[1] = z*ws2[0] - x*ws2[2];
                hcol2[2] = x*ws2[1] - y*ws2[0];

                // fill H_rr (row-major 6x6)
                local_H[0*6+0] = hcol0[0]; local_H[0*6+1] = hcol1[0]; local_H[0*6+2] = hcol2[0];
                local_H[1*6+0] = hcol0[1]; local_H[1*6+1] = hcol1[1]; local_H[1*6+2] = hcol2[1];
                local_H[2*6+0] = hcol0[2]; local_H[2*6+1] = hcol1[2]; local_H[2*6+2] = hcol2[2];

                // H_tt = Omega
                local_H[3*6+3] = Omega[0]; local_H[3*6+4] = Omega[1]; local_H[3*6+5] = Omega[2];
                local_H[4*6+3] = Omega[3]; local_H[4*6+4] = Omega[4]; local_H[4*6+5] = Omega[5];
                local_H[5*6+3] = Omega[6]; local_H[5*6+4] = Omega[7]; local_H[5*6+5] = Omega[8];

                // H_rt (rot,trans): rows are ws0/ws1/ws2 (Rustと一致)
                local_H[0*6+3] = ws0[0]; local_H[0*6+4] = ws0[1]; local_H[0*6+5] = ws0[2];
                local_H[1*6+3] = ws1[0]; local_H[1*6+4] = ws1[1]; local_H[1*6+5] = ws1[2];
                local_H[2*6+3] = ws2[0]; local_H[2*6+4] = ws2[1]; local_H[2*6+5] = ws2[2];

                // H_tr = transpose(H_rt)
                local_H[3*6+0] = local_H[0*6+3];
                local_H[3*6+1] = local_H[1*6+3];
                local_H[3*6+2] = local_H[2*6+3];

                local_H[4*6+0] = local_H[0*6+4];
                local_H[4*6+1] = local_H[1*6+4];
                local_H[4*6+2] = local_H[2*6+4];

                local_H[5*6+0] = local_H[0*6+5];
                local_H[5*6+1] = local_H[1*6+5];
                local_H[5*6+2] = local_H[2*6+5];
            }
        }
    }

    // ---- block reduction via shared memory (your style) ----
    __shared__ float s_H[36];
    __shared__ float s_b[6];

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i=0;i<36;i++) s_H[i] = 0.0f;
#pragma unroll
        for (int i=0;i<6;i++)  s_b[i] = 0.0f;
    }
    __syncthreads();

    if (idx < num_source) {
#pragma unroll
        for (int i=0;i<36;i++) atomicAdd(&s_H[i], local_H[i]);
#pragma unroll
        for (int i=0;i<6;i++)  atomicAdd(&s_b[i], local_b[i]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i=0;i<36;i++) atomicAdd(&d_H[i], s_H[i]);
#pragma unroll
        for (int i=0;i<6;i++)  atomicAdd(&d_b[i], s_b[i]);
    }
}
