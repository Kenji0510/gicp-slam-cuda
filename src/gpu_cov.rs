use std::{any, sync::Arc};
use anyhow::{Result, Context};
use cudarc::{driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
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

    pub fn compute_covariances<'a>(
        &'a mut self,
        d_points: &CudaSlice<f32>,
        num_points: usize,
        is_target: bool,
    // ) -> Result<Vec<MatrixXx3<f64>>> {
    ) -> Result<(CudaView<'a, f32>, Vec<Matrix3<f64>>)> {
        if num_points == 0 {
            anyhow::bail!("No points provided for covariance computation");
        }

        // Self::ensure_buffer(&self.stream, &mut self.buf_points, num_points * 3)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_covs, num_points * 9)?;

        // let d_points = self.buf_points.as_mut().unwrap();
        let d_covs = self.buf_covs.as_mut().unwrap();

        // let mut d_points_view = d_points.slice_mut(0..num_points * 3);
        // let mut d_covs_view = d_covs.slice_mut(0..num_points * 9);

        // self.stream.memcpy_htod(points_slice, &mut d_points_view)?;

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

        if is_target {
            let raw_covs = self.stream.clone_dtoh(&d_covs_view)
            .context("Failed DtoH copy for covariances")?;

            let result: Vec<Matrix3<f64>> = raw_covs
                .par_chunks(9)
                .map(|chunk| {
                    Matrix3::new(
                        chunk[0] as f64, chunk[1] as f64, chunk[2] as f64,
                        chunk[3] as f64, chunk[4] as f64, chunk[5] as f64,
                        chunk[6] as f64, chunk[7] as f64, chunk[8] as f64,
                    )
                })
                .collect();

            Ok((d_covs_view, result))
        } else {
            Ok((d_covs_view, vec![]))
        }
        // Ok(result)
    }
}