use std::sync::Arc;
use anyhow::{Result, Context};
use cudarc::{driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use nalgebra::{Matrix3, MatrixXx3, SymmetricEigen};
use ndarray::Array2;
use rayon::prelude::*;


pub struct CudaCovContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,

    buf_points: Option<CudaSlice<f32>>,
    buf_covs: Option<CudaSlice<f32>>,
}

impl CudaCovContext {
    pub fn new(ptx_path: &str) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        Self::from_context(ctx, ptx_path)
    }

    pub fn from_context(
        ctx: Arc<CudaContext>, 
        ptx_path: &str,
    ) -> Result<Self> {
        let stream = ctx.default_stream();

        let module = ctx.load_module(Ptx::from_file(ptx_path))
            .context("Failed to load PTX module")?;

        let func = module.load_function("compute_covariance")
            .context("Kernel function not found in PTX")?;

        Ok(Self {
            ctx, 
            stream,
            func,
            buf_points: None,
            buf_covs: None,
        })
    }

    fn ensure_buffer(
        stream: &Arc<CudaStream>,
        buffer: &mut Option<CudaSlice<f32>>,
        required_len: usize,
    ) -> Result<()> {
        let current_cap = buffer.as_ref().map(|b| b.len()).unwrap_or(0);
        if current_cap < required_len {
            let new_cap = (required_len as f32 * 1.2) as usize;
            *buffer = Some(stream.alloc_zeros::<f32>(new_cap)?);
        }
        Ok(())
    }

    pub fn compute_covariances(
        &mut self,
        points: &Array2<f32>,
    // ) -> Result<Vec<MatrixXx3<f64>>> {
    ) -> Result<Vec<Matrix3<f64>>> {
        let num_points = points.nrows();
        if num_points == 0 {
            return Ok(vec![]);
        }

        Self::ensure_buffer(&self.stream, &mut self.buf_points, num_points * 3)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_covs, num_points * 9)?;

        let d_points = self.buf_points.as_mut().unwrap();
        let d_covs = self.buf_covs.as_mut().unwrap();

        let points_slice = points.as_slice()
            .context("Points not contiguous")?;

        let mut d_points_view = d_points.slice_mut(0..num_points * 3);
        // let mut d_covs_view = d_covs.slice_mut(0..num_points * 9);

        self.stream.memcpy_htod(points_slice, &mut d_points_view)?;

        let cfg = LaunchConfig::for_num_elems(num_points as u32);

        // let start = std::time::Instant::now();
        unsafe {
            self.stream.launch_builder(&self.func)
                .arg(d_points)
                .arg(&(num_points as i32))
                .arg(d_covs)
                .launch(cfg)
                .context("Kernel launch failed")?;
        }

        self.stream.synchronize()
            .context("Stream sync failed")?;

        // let _duration = start.elapsed();
        // println!("Calculating covariance took: {:?}", _duration);

        let d_covs = self.buf_covs.as_ref().unwrap();
        let d_covs_view = d_covs.slice(0..num_points * 9);

        let raw_covs = self.stream.clone_dtoh(&d_covs_view)
            .context("Failed DtoH copy for covariances")?;

        // let mut result = Vec::with_capacity(num_points);
        // for i in 0..num_points {
        //     let offset = i * 9;
        //     let cov = Matrix3::from_row_slice(&[
        //         raw_covs[offset] as f64,     raw_covs[offset + 1] as f64, raw_covs[offset + 2] as f64,
        //         raw_covs[offset + 3] as f64, raw_covs[offset + 4] as f64, raw_covs[offset + 5] as f64,
        //         raw_covs[offset + 6] as f64, raw_covs[offset + 7] as f64, raw_covs[offset + 8] as f64,
        //     ]);
        //     result.push(cov);
        // }

        let regularized_covs: Vec<Matrix3<f64>> = raw_covs
            .par_chunks(9) // 3x3=9要素ずつ処理
            .map(|chunk| {
                // f32 -> f64 へ変換
                let mat = Matrix3::new(
                    chunk[0] as f64, chunk[1] as f64, chunk[2] as f64,
                    chunk[3] as f64, chunk[4] as f64, chunk[5] as f64,
                    chunk[6] as f64, chunk[7] as f64, chunk[8] as f64,
                );

                // 固有値分解
                let eigen = SymmetricEigen::new(mat);
                let rot = eigen.eigenvectors;
                let mut vals = eigen.eigenvalues;

                // 固有値をソートして正規化 (平面性を強調)
                // GICP Regularization: min_eigen = 1e-3
                let mut pairs: Vec<(f64, usize)> = vals.iter()
                    .cloned()
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect();
                // 昇順ソート
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // 最小の固有値を潰して「平ら」にする
                vals[pairs[0].1] = 1e-3; 
                vals[pairs[1].1] = 1.0;
                vals[pairs[2].1] = 1.0;

                // 再構築: R * S * R^T
                rot * Matrix3::from_diagonal(&vals) * rot.transpose()
            })
            .collect();
        
        // let elapsed = start.elapsed();
        // println!("Regularizing covariances took: {:?}", elapsed);

        Ok(regularized_covs)

        // Ok(result)
    }
}