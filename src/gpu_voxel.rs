use std::sync::Arc;
use anyhow::{Result, Context};
use cudarc::{driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use ndarray::Array2;



pub struct CudaVoxelContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func_init: CudaFunction,
    func_insert: CudaFunction,
    func_compact: CudaFunction,

    buf_table_keys: Option<CudaSlice<u64>>,
    buf_table_centroids: Option<CudaSlice<f32>>,
    buf_table_counts: Option<CudaSlice<i32>>,

    buf_input_points: Option<CudaSlice<f32>>,
    buf_out_points: Option<CudaSlice<f32>>,
    buf_valid_count: Option<CudaSlice<i32>>,
}

impl CudaVoxelContext {
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        let stream = ctx.default_stream();

        let module = ctx.load_module(Ptx::from_file(ptx_path))
            .context("Failed to load PTX module")?;

        Ok(Self {
            ctx,
            stream,
            func_init: module.load_function("init_table")?,
            func_insert: module.load_function("insert_points")?,
            func_compact: module.load_function("compact_voxels")?,
            buf_table_keys: None,
            buf_table_centroids: None,
            buf_table_counts: None,
            buf_input_points: None,
            buf_out_points: None,
            buf_valid_count: None,
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

    pub fn voxel_downsample<'a>(
        &'a mut self,
        input_points: &Array2<f32>,
        num_points: usize,
        voxel_size: f32,
        is_target: bool,
    ) -> Result<(CudaView<'a, f32>, usize, Array2<f32>)> {
        if num_points == 0 {
            anyhow::bail!("No input points for voxel downsampling");
        }

        let table_size = num_points * 2;

        Self::ensure_buffer(&self.stream, &mut self.buf_table_keys, table_size)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_table_centroids, table_size * 3)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_table_counts, table_size)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_input_points, num_points * 3)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_out_points, num_points * 3)?;
        Self::ensure_buffer(&self.stream, &mut self.buf_valid_count, 1)?;

        // Scope the mutable borrows so they drop before we need immutable access
        {
            let d_keys = self.buf_table_keys.as_mut().unwrap();
            let d_centroids = self.buf_table_centroids.as_mut().unwrap();
            let d_counts = self.buf_table_counts.as_mut().unwrap();
            let d_input_points = self.buf_input_points.as_mut().unwrap();
            let d_out_points = self.buf_out_points.as_mut().unwrap();
            let d_counter = self.buf_valid_count.as_mut().unwrap();

            let input_slice = input_points.as_slice()
                .context("Input points not contiguous")?;
            self.stream.memcpy_htod(input_slice, &mut d_input_points.slice_mut(0..num_points * 3))?;

            self.stream.memcpy_htod(&[0i32], &mut d_counter.slice_mut(0..1))?;
            self.stream.memset_zeros(d_centroids)?;
            self.stream.memset_zeros(d_counts)?;

            let init_cfg = LaunchConfig::for_num_elems(table_size as u32);
            unsafe {
                self.stream.launch_builder(&self.func_init)
                    .arg(&d_keys.slice(0..table_size))
                    .arg(&(table_size as i32))
                    .launch(init_cfg)
                    .context("Failed to launch init_table kernel")?;
            }

            let insert_cfg = LaunchConfig::for_num_elems(num_points as u32);
            unsafe {
                self.stream.launch_builder(&self.func_insert)
                    .arg(&d_input_points.slice(0..num_points * 3))
                    .arg(&(num_points as i32))
                    .arg(&voxel_size)
                    .arg(&d_keys.slice(0..table_size))
                    .arg(&d_centroids.slice(0..table_size * 3))
                    .arg(&d_counts.slice(0..table_size))
                    .arg(&(table_size as i32))
                    .launch(insert_cfg)
                    .context("Failed to launch insert_points kernel")?;
            }

            let compact_cfg = LaunchConfig::for_num_elems(table_size as u32);
            unsafe {
                self.stream.launch_builder(&self.func_compact)
                    .arg(&d_keys.slice(0..table_size))
                    .arg(&d_centroids.slice(0..table_size * 3))
                    .arg(&d_counts.slice(0..table_size))
                    .arg(&(table_size as i32))
                    .arg(&d_out_points.slice(0..num_points * 3))
                    .arg(&d_counter.slice(0..1))
                    .launch(compact_cfg)
                    .context("Failed to launch compact_table kernel")?;
            }
        } // All mutable borrows dropped here

        self.stream.synchronize()
            .context("Stream sync failed after voxel downsampling")?;

        if is_target {
            // Now we can safely create immutable borrows
            let d_counter = self.buf_valid_count.as_ref().unwrap();
            let valid_count_vec = self.stream.clone_dtoh(&d_counter.slice(0..1))?;
            let valid_count = valid_count_vec[0] as usize;

            let d_out_points = self.buf_out_points.as_ref().unwrap();
            let out_pts_host = self.stream.clone_dtoh(
                &d_out_points.slice(0..valid_count * 3)
            )?;

            let out_pts_array = Array2::from_shape_vec(
                (valid_count, 3),
                out_pts_host,
            ).context("Failed to create output points array")?;

            Ok((d_out_points.slice(0..valid_count * 3), valid_count, out_pts_array))
        } else {
            let d_counter = self.buf_valid_count.as_ref().unwrap();
            let valid_count_vec = self.stream.clone_dtoh(&d_counter.slice(0..1))?;
            let valid_count = valid_count_vec[0] as usize;

            Ok((self.buf_out_points.as_ref().unwrap().slice(0..valid_count * 3), valid_count, Array2::<f32>::zeros((0,0))))
        }
    }
}