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

    pub fn apply<'a>(
        &'a mut self,
        src_points: &CudaSlice<f32>,
        src_covs: &CudaSlice<f32>,
        num_points: usize,
        transform: &Array2<f32>,
    ) -> Result<(CudaView<'a, f32>, CudaView<'a, f32>, Array2<f32>, Vec<Matrix3<f64>>)> {
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

        let out_pts = self.buf_out_points.as_ref().unwrap();
        let out_covs = self.buf_out_covs.as_ref().unwrap();

        // Copy d to host
        let transformed_points_vec = self.stream.clone_dtoh(
            &out_pts.slice(0..num_points * 3)
        )?;
        let transformed_covs_vec = self.stream.clone_dtoh(
            &out_covs.slice(0..num_points * 9)
        )?;

        let transformed_points = Array2::from_shape_vec(
            (num_points, 3), 
            transformed_points_vec
        ).context("Failed to reshape transformed points")?;

        // Convert covariances to Vec<Matrix3<f64>>
        let transformed_covs: Vec<Matrix3<f64>> = transformed_covs_vec
            .par_chunks(9)
            .map(|chunk| {
                Matrix3::new(
                    chunk[0] as f64, chunk[1] as f64, chunk[2] as f64,
                    chunk[3] as f64, chunk[4] as f64, chunk[5] as f64,
                    chunk[6] as f64, chunk[7] as f64, chunk[8] as f64,
                )
            })
            .collect();

        Ok((
            out_pts.slice(0..num_points * 3),
            out_covs.slice(0..num_points * 9),
            transformed_points,
            transformed_covs,
        ))
    }
}