use core::f64;
use std::collections::VecDeque;

use anyhow::{Result, Context};
use cudarc::driver::CudaContext;
use gicp_slam_cuda::{gpu_cov::CudaCovContext, gpu_search::CudaKnnContext, load_files::{load_and_flatten_imu_json, load_pcd_files}, operate_pcd_file::load_pcd_xyzt, pre_process_pcd::{self, preprocess_point_cloud}, predict_pose_imu::{self, build_rotation_trajectory, predict_pose_by_imu}};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use ndarray::{Array2, Axis};
use serde::Serialize;


const PCD_DIR: &str = "data/input/mid360/pcds/mid360-20251125-03";
const IMU_FILE_PATH: &str = "data/input/mid360/imu/mid360-imu-20251125-03/imu_data.json";

const KNN_PTX_PATH: &str = "src/kernels/search.ptx";
const COV_PTX_PATH: &str = "src/kernels/compute_covariance.ptx";

const MIN_DIST: f32 = 0.0;
const MAX_DIST: f32 = 20.0;
const MAX_ITERATIONS: usize = 10;

#[derive(Serialize)]
struct PoseData {
    timestamp: f64,
    position: [f32; 3],      // x, y, z
    rotation_quat: [f32; 4], // w, x, y, z
}

struct FrameData {
    points: Array2<f32>,
    covariances: Vec<Matrix3<f64>>,
}

#[derive(Serialize)]
struct RmseStats {
    frame_index: usize,
    rmse: f32,
}

struct GicpOdometry {
    // Global pose as 4x4 transformation matrix
    current_g_pose: Array2<f32>,
    velocity: Vector3<f64>,
    last_timestamp: f64,
    local_map: VecDeque<FrameData>,
    global_map: Vec<Array2<f32>>,
    gicp_traj_log: Vec<PoseData>,
    rmse_log: Vec<RmseStats>,
}

struct ProcessTimeStats {
    total_gicp_time: std::time::Duration,
}

fn main() -> Result<()> {
    // Load PCD files and IMU data
    let pcd_paths = load_pcd_files(PCD_DIR)
        .expect("Failed to load PCD_DIR");
    let imu_data = load_and_flatten_imu_json(IMU_FILE_PATH)
        .expect("Failed to load imu data");

    println!("Loaded {} PCD files from {}", pcd_paths.len(), PCD_DIR);
    println!("Loaded {} IMU samples from {}", imu_data.len(), IMU_FILE_PATH);

    let mut icp_trajectory_log: Vec<PoseData> = Vec::new();

    let init_pcd = load_pcd_xyzt(pcd_paths[0].to_str().unwrap())
        .expect("Failed to load initial PCD file");

    println!("Initializing CUDA...");
    let ctx = CudaContext::new(0)
        .context("Failed to create CUDA context")?;
    let mut gpu_knn = CudaKnnContext::from_context(ctx.clone(), KNN_PTX_PATH)?;
    let mut gpu_cov = CudaCovContext::from_context(ctx.clone(), COV_PTX_PATH)?;
    println!("CUDA initialized.");

    let mut gicp_odometry = GicpOdometry {
        current_g_pose: Array2::eye(4),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        last_timestamp: 0.0,
        local_map: VecDeque::new(),
        global_map: Vec::new(),
        gicp_traj_log: Vec::new(),
        rmse_log: Vec::new(),
    };

    let mut process_time_stats = ProcessTimeStats {
        total_gicp_time: std::time::Duration::new(0, 0),
    };

    // Initialize global map accumulator
    let mut global_map_accumulator = Vec::new();
    global_map_accumulator.push(init_pcd);

    let base_timestamp = imu_data[0].timestamp_sec;
    gicp_odometry.gicp_traj_log.push(
        extract_pose_from_matrix(base_timestamp, &gicp_odometry.current_g_pose)
    );

    for (i, pcd_path) in pcd_paths.iter().enumerate().skip(1) {
        // Load current pcd frame
        let pcd_points = load_pcd_xyzt(pcd_path.to_str().unwrap())
            .expect("Failed to load pcd data");

        let min_timestamp = pcd_points.iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);

        // Transform the min_timestamp to seconds
        let current_frame_timestamp = min_timestamp / 1_000_000_000.0;

        // Predict pose using IMU data
        let (predicted_pose, predicted_velocity) = predict_pose_by_imu(
            &gicp_odometry.current_g_pose, 
            &gicp_odometry.velocity, 
            gicp_odometry.last_timestamp, 
            current_frame_timestamp, 
            &imu_data
        );

        let min_timestamp = pcd_points.iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let max_timestamp = pcd_points.iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        // Build rotation trajectory from IMU data
        let rotation_traj = build_rotation_trajectory(
            &imu_data, 
            gicp_odometry.last_timestamp, 
            current_frame_timestamp
        );

        // Preprocess point cloud: deskewing and filtering
        let preprocessed_current_points = preprocess_point_cloud(
            &pcd_points, 
            &rotation_traj, 
            MIN_DIST, 
            MAX_DIST
        );

        let (target_pts, target_covs) = if gicp_odometry.local_map.is_empty() {
            (preprocessed_current_points.clone(), vec![])
        } else {
            flatten_local_map(&gicp_odometry.local_map)?
        };

        // Debug
        let voxel_size = 0.5;
        let (v_preprocessed_current_points, v_preprocessed_current_covs) = voxel_downsample_with_cov(&preprocessed_current_points, &vec![], voxel_size);
        let (v_target_pts, v_target_covs) = voxel_downsample_with_cov(&target_pts, &target_covs, voxel_size);

        // println!("Debug: source points: {}, target points: {}", preprocessed_current_points.nrows(), target_pts.nrows());
        println!("Debug: v_source points: {}, v_target points: {}", v_preprocessed_current_points.nrows(), v_target_pts.nrows());
        
        // Compute covariances for current frame points
        let computed_covs = gpu_cov.compute_covariances(&v_preprocessed_current_points)
            .expect("Failed to compute covariances on GPU");

        let mut current_transform = predicted_pose.clone();

        let start = std::time::Instant::now();
        for i in 0..MAX_ITERATIONS {
            // Rotation source points by predicted pose
            let transformd_source = transform_points(&v_preprocessed_current_points, &current_transform);

            // let (indices, dists_sq) = gpu_knn.find_nearest(&preprocessed_current_points, &target_pts)
            let (indices, dists_sq) = gpu_knn.find_nearest(&transformd_source, &v_target_pts)
                .expect("Failed to perform GPU k-NN search");


        }
        let gicp_duration = start.elapsed();
        println!("GICP for frame {} took {:?}", i, gicp_duration);

        
    }

    Ok(())
}

fn extract_pose_from_matrix(timestamp: f64, transform: &Array2<f32>) -> PoseData {
    let tx = transform[[0, 3]];
    let ty = transform[[1, 3]];
    let tz = transform[[2, 3]];

    // 回転行列成分を抽出
    let mat3 = Matrix3::new(
        transform[[0, 0]] as f64, transform[[0, 1]] as f64, transform[[0, 2]] as f64,
        transform[[1, 0]] as f64, transform[[1, 1]] as f64, transform[[1, 2]] as f64,
        transform[[2, 0]] as f64, transform[[2, 1]] as f64, transform[[2, 2]] as f64,
    );
    let q = UnitQuaternion::from_matrix(&mat3);

    PoseData {
        timestamp,
        position: [tx, ty, tz],
        rotation_quat: [q.w as f32, q.i as f32, q.j as f32, q.k as f32],
    }
}

fn flatten_local_map(
    queue: &VecDeque<FrameData>
) -> Result<(Array2<f32>, Vec<Matrix3<f64>>)> {
    
    // 1. 点群 (Array2) の結合
    let points_views: Vec<_> = queue.iter()
        .map(|frame| frame.points.view())
        .collect();

    // Axis(0) = 行方向（縦）に結合
    let merged_points = ndarray::concatenate(Axis(0), &points_views)
        .context("Failed to concatenate local map points")?;

    // 2. 共分散 (Vec) の結合
    let total_points = merged_points.nrows();
    let mut merged_covs = Vec::with_capacity(total_points);

    for frame in queue {
        merged_covs.extend_from_slice(&frame.covariances);
    }

    // 整合性チェック (念のため)
    if merged_points.nrows() != merged_covs.len() {
        return Err(anyhow::anyhow!(
            "Mismatch between points count ({}) and covariances count ({}) in local map",
            merged_points.nrows(),
            merged_covs.len()
        ));
    }

    Ok((merged_points, merged_covs))
}

fn voxel_downsample_with_cov(
    pts: &Array2<f32>,
    covs: &[Matrix3<f64>],
    voxel_size: f32
) -> (Array2<f32>, Vec<Matrix3<f64>>) {
    let mut grid = std::collections::HashMap::new();

    let default_cov = Matrix3::<f64>::identity();

    for i in 0..pts.nrows() {
        let x = pts[[i, 0]];
        let y = pts[[i, 1]];
        let z = pts[[i, 2]];
        let p_vec = Vector3::new(x, y, z);

        let ix = (x / voxel_size).floor() as i32;
        let iy = (y / voxel_size).floor() as i32;
        let iz = (z / voxel_size).floor() as i32;
        let key = (ix, iy, iz);

        let cov = if i < covs.len() {
            covs[i]
        } else {
            default_cov
        };

        grid.entry(key)
            // 既にボクセルに点がある場合：座標を足し合わせ、カウントを増やす
            .and_modify(|(sum, count, _)| {
                *sum += p_vec;
                *count += 1;
            })
            // 初めての点の場合：座標、カウント1、そして共分散を保存
            .or_insert((p_vec, 1, cov));
    }

    // 抽出（重心を計算）
    let n_kept = grid.len();
    let mut new_pts = Array2::<f32>::zeros((n_kept, 3));
    let mut new_covs = Vec::with_capacity(n_kept);

    for (k, (_, (sum, count, first_cov))) in grid.iter().enumerate() {
        // 重心 = 合計 / 個数
        let centroid = sum / (*count as f32);
        
        new_pts[[k, 0]] = centroid.x;
        new_pts[[k, 1]] = centroid.y;
        new_pts[[k, 2]] = centroid.z;
        
        // 共分散は「そのボクセルを代表する鋭い分布」として、最初の点のものを採用
        new_covs.push(*first_cov);
    }

    (new_pts, new_covs)
}

/// Transform 3D points using ndarray operations
fn transform_points(points: &Array2<f32>, transform: &Array2<f32>) -> Array2<f32> {
    let n = points.nrows();
    
    // Extract 3×3 rotation matrix
    let rotation = transform.slice(ndarray::s![0..3, 0..3]);
    
    // Extract translation vector (3×1)
    let translation = transform.slice(ndarray::s![0..3, 3]);
    
    // result = points @ R^T + t
    // (N×3) @ (3×3) + (3,) broadcasts to (N×3)
    let rotated = points.dot(&rotation.t());
    
    // Add translation using broadcasting
    rotated + &translation.insert_axis(Axis(0))
}