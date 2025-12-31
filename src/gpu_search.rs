use std::sync::Arc;
use anyhow::{Result, Context};
use cudarc::{driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use ndarray::Array2;



pub struct CudaKnnContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,

    buf_indices: Option<CudaSlice<i32>>,
    buf_dists: Option<CudaSlice<f32>>,
}

impl CudaKnnContext {
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        let stream = ctx.default_stream();

        let module = ctx.load_module(Ptx::from_file(ptx_path))
            .context("Failed to load PTX module")?;

        Ok(Self {
            ctx,
            stream,
            func: module.load_function("find_nearest_neighbor")?,
            buf_indices: None,
            buf_dists: None,
        })
    }

    fn ensure_buffer<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        stream: &Arc<CudaStream>,
        buffer: &mut Option<CudaSlice<T>>,
        len: usize,
    ) -> Result<()> {
        let current = buffer.as_ref()
            .map(|b| b.len())
            .unwrap_or(0);
        if current < len {
            let new_len = (len as f32 * 1.2) as usize;
            *buffer = Some(stream.alloc_zeros::<T>(new_len)?);
        }
        Ok(())
    }

    pub fn find_nearest(
        &mut self,
        d_source_pts: &CudaView<f32>,
        num_source: usize,
        d_target_pts: &CudaView<f32>,
        num_target: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<f32>, Vec<i32>, Vec<f32>)> {
        // let num_source = d_source_pts.nrows();
        // let num_target = d_target_pts.nrows();

        if num_source == 0 || num_target == 0 {
            anyhow::bail!("Empty point cloud");
        }

        Self::ensure_buffer(&self.stream, &mut self.buf_indices, num_source)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_dists, num_source)?;

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
        // let mut d_indices: CudaSlice<i32> = self.stream
        //     .alloc_zeros(num_source)
        //     .context("Failed to alloc d_indices")?;
        let d_indices = self.buf_indices.as_mut().unwrap();
        // let mut d_distances: CudaSlice<f32> = self.stream
        //     .alloc_zeros(num_source)
        //     .context("Failed to alloc d_distances")?;
        let d_distances = self.buf_dists.as_mut().unwrap();

        let cfg = LaunchConfig::for_num_elems(num_source as u32);

        // let start = std::time::Instant::now();
        unsafe {
            self.stream.launch_builder(&self.func)
            .arg(d_source_pts)
            .arg(d_target_pts)
            .arg(&(num_source as i32))
            .arg(&(num_target as i32))
            .arg(d_indices)
            .arg(d_distances)
            .launch(cfg)
            .context("Kernel launch failed")?;
        }

        self.stream.synchronize()
            .context("Stream sync failed")?;

        // let _duration = start.elapsed();
        // println!("KNN search took: {:?}", _duration);

        let d_indices = self.buf_indices.as_ref().unwrap().clone();
        let d_distances = self.buf_dists.as_ref().unwrap().clone();

        let indices: Vec<i32> = self.stream
            .clone_dtoh(&d_indices)
            .context("Failed DtoH copy(indices")?;
        let distances: Vec<f32> = self.stream
            .clone_dtoh(&d_distances)
            .context("Failed DtoH copy(distances")?;

        // Ok((indices, distances))

        Ok((d_indices, d_distances, indices, distances))
    }
}