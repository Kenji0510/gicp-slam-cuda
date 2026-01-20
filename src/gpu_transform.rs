use std::sync::Arc;
use anyhow::{Result, Context};
use cudarc::{driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use nalgebra::Matrix3;
use ndarray::Array2;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};



pub struct CudaTransformContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,

    buf_out_points: Option<CudaSlice<f32>>,
    buf_out_covs: Option<CudaSlice<f32>>,
}

impl CudaTransformContext {
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_file(ptx_path))
            .context("Failed to load Transform PTX")?;
        let func = module.load_function("transform_points_and_covs")?;

        Ok(Self {
            ctx,
            stream,
            func,
            buf_out_points: None,
            buf_out_covs: None,
        })
    }

    fn ensure_buffer(
        stream: &Arc<CudaStream>,
        buffer: &mut Option<CudaSlice<f32>>,
        len: usize
    ) -> Result<()> {
        let current = buffer.as_ref().map(|b| b.len()).unwrap_or(0);
        if current < len {
            let new_len = (len as f32 * 1.2) as usize; 
            *buffer = Some(stream.alloc_zeros::<f32>(new_len)?);
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        src_points: &CudaView<f32>,
        src_covs: &CudaView<f32>,
        num_points: usize,
        transform: &Array2<f32>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        Self::ensure_buffer(&self.stream, &mut self.buf_out_points, num_points * 3)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_out_covs, num_points * 9)?;

        let out_pts = self.buf_out_points.as_mut().unwrap();
        let out_covs = self.buf_out_covs.as_mut().unwrap();

        let mut out_pts_view = out_pts.slice_mut(0..num_points * 3);
        let mut out_covs_view = out_covs.slice_mut(0..num_points * 9);

        let r00 = transform[[0,0]]; let r01 = transform[[0,1]]; let r02 = transform[[0,2]]; let t0 = transform[[0,3]];
        let r10 = transform[[1,0]]; let r11 = transform[[1,1]]; let r12 = transform[[1,2]]; let t1 = transform[[1,3]];
        let r20 = transform[[2,0]]; let r21 = transform[[2,1]]; let r22 = transform[[2,2]]; let t2 = transform[[2,3]];

        let cfg = LaunchConfig::for_num_elems(num_points as u32);
        unsafe {
            self.stream.launch_builder(&self.func)
                .arg(src_points)
                .arg(src_covs)
                .arg(&(num_points as i32))
                .arg(&r00).arg(&r01).arg(&r02).arg(&t0)
                .arg(&r10).arg(&r11).arg(&r12).arg(&t1)
                .arg(&r20).arg(&r21).arg(&r22).arg(&t2)
                .arg(&mut out_pts_view)
                .arg(&mut out_covs_view)
                .launch(cfg)
                .context("Failed to launch transform kernel")?;
        }

        self.stream.synchronize()
            .context("Failed to synchronize stream after transform")?;

        let out_pts = self.buf_out_points.as_ref().unwrap().clone();
        let out_covs = self.buf_out_covs.as_ref().unwrap().clone();

        Ok((out_pts, out_covs))
    }
}