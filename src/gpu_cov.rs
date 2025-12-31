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
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        let stream = ctx.default_stream();

        let module = ctx.load_module(Ptx::from_file(ptx_path))
            .context("Failed to load PTX module")?;
        
        
        Ok(Self {
            ctx,
            stream,
            func: module.load_function("compute_covariance")?,
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
        d_points: &CudaView<f32>,
        num_points: usize,
    ) -> Result<CudaSlice<f32>> {
        if num_points == 0 {
            anyhow::bail!("No points provided for covariance computation");
        }

        Self::ensure_buffer(&self.stream, &mut self.buf_covs, num_points * 9)?;

        let d_covs = self.buf_covs.as_mut().unwrap();

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

        // self.stream.synchronize()
        //     .context("Stream sync failed")?;

        // let _duration = start.elapsed();
        // println!("Calculating covariance took: {:?}", _duration);

        let d_covs = self.buf_covs.as_ref().unwrap().clone();

        Ok(d_covs)
    }
}