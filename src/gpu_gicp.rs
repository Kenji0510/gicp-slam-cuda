use std::sync::Arc;
use anyhow::{Result, Context};
use cudarc::{driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use ndarray::{Array1, Array2};



pub struct CudaGicpContext {
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl CudaGicpContext {
    pub fn new(ptx_path: &str) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        Self::from_context(ctx, ptx_path)
    }

    pub fn from_context(ctx: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_file(ptx_path))?;
        let func = module.load_function("compute_gicp_linear_system")?;
        Ok(Self { stream, func })
    }

    pub fn compute_gicp(
        &self,
        d_source_pts: &CudaView<f32>,
        d_source_covs: &CudaView<f32>,
        d_target_pts: &CudaView<f32>,
        d_target_covs: &CudaView<f32>,
        d_indices: &CudaSlice<i32>,
        d_distances: &CudaSlice<f32>,
        max_dist_sq: f32,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let num_source = d_source_pts.len() / 3;
        let num_target = d_target_pts.len() / 3;

        if num_source == 0 || num_target == 0 {
            anyhow::bail!("Empty point cloud");
        }
        
        let mut d_h: CudaSlice<f32> = self.stream
            .alloc_zeros(36)
            .context("Failed to alloc d_h")?;
        let mut d_b: CudaSlice<f32> = self.stream
            .alloc_zeros(6)
            .context("Failed to alloc d_b")?;

        let cfg = LaunchConfig::for_num_elems(num_source as u32);

        unsafe {
            self.stream.launch_builder(&self.func)
                .arg(d_source_pts)
                .arg(d_source_covs)
                .arg(d_target_pts)
                .arg(d_target_covs)
                .arg(d_indices)
                .arg(d_distances)
                .arg(&(num_source as i32))
                .arg(&(num_target as i32))
                .arg(&max_dist_sq)
                .arg(&mut d_h)
                .arg(&mut d_b)
                .launch(cfg)
                .context("GICP Kernel launch failed")?;
        }

        self.stream.synchronize()
            .context("Stream sync failed after voxel downsampling")?;

        // 3. 結果を CPU にコピー (非常に小さいデータなので高速)
        let h_vec_f32: Vec<f32> = self.stream.clone_dtoh(&d_h)?;
        let b_vec_f32: Vec<f32> = self.stream.clone_dtoh(&d_b)?;

        // 4. ndarray (f64) に変換して返す
        // ※ GPU計算が f32 でも、ソルバーの数値安定性のために f64 にキャストするのが一般的です
        let h_vec_f64: Vec<f64> = h_vec_f32.into_iter().map(|v| v as f64).collect();
        let b_vec_f64: Vec<f64> = b_vec_f32.into_iter().map(|v| v as f64).collect();

        // 6x6 行列と 6要素ベクトルに整形
        let h_matrix = Array2::from_shape_vec((6, 6), h_vec_f64)
            .context("Failed to reshape H matrix")?;
        let b_vector = Array1::from_vec(b_vec_f64);

        Ok((h_matrix, b_vector))
    }
}