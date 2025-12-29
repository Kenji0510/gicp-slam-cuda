use std::sync::Arc;
use anyhow::{Result, Context};
use cudarc::{driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use ndarray::Array2;



pub struct CudaKnnContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl CudaKnnContext {
    pub fn new(ptx_path: &str) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        Self::from_context(ctx, ptx_path)
    }

    pub fn from_context(ctx: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_file(ptx_path))?;
        let func = module.load_function("find_nearest_neighbor")?;
        Ok(Self { ctx, stream, func })
    }

    pub fn find_nearest(
        &self,
        d_source_pts: &CudaSlice<f32>,
        num_source: usize,
        d_target_pts: &CudaSlice<f32>,
        num_target: usize,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        // let num_source = d_source_pts.nrows();
        // let num_target = d_target_pts.nrows();

        if num_source == 0 || num_target == 0 {
            anyhow::bail!("Empty point cloud");
        }

        // let source_slice = d_source_pts.as_slice()
        //     .context("Source not contignous")?;
        // let target_slice = d_target_pts.as_slice()
        //     .context("Target not contignous")?;

        // let d_source: CudaSlice<f32> = self.stream
        //     .clone_htod(source_slice)
        //     .context("Failed HtoD copy (source)")?;
        // let d_target: CudaSlice<f32> = self.stream
        //     .clone_htod(target_slice)
        //     .context("Failed HtoD copy (target)")?;
        let mut d_indices: CudaSlice<i32> = self.stream
            .alloc_zeros(num_source)
            .context("Failed to alloc d_indices")?;
        let mut d_distances: CudaSlice<f32> = self.stream
            .alloc_zeros(num_source)
            .context("Failed to alloc d_distances")?;

        let cfg = LaunchConfig::for_num_elems(num_source as u32);

        // let start = std::time::Instant::now();
        unsafe {
            self.stream.launch_builder(&self.func)
            .arg(d_source_pts)
            .arg(d_target_pts)
            .arg(&(num_source as i32))
            .arg(&(num_target as i32))
            .arg(&mut d_indices)
            .arg(&mut d_distances)
            .launch(cfg)
            .context("Kernel launch failed")?;
        }

        self.stream.synchronize()
            .context("Stream sync failed")?;

        // let _duration = start.elapsed();
        // println!("KNN search took: {:?}", _duration);

        let indices: Vec<i32> = self.stream
            .clone_dtoh(&d_indices)
            .context("Failed DtoH copy(indices")?;
        let distances: Vec<f32> = self.stream
            .clone_dtoh(&d_distances)
            .context("Failed DtoH copy(distances")?;

        Ok((indices, distances))
    }
}