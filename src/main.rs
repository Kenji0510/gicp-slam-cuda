use core::f64;
use std::{collections::VecDeque, f32::{INFINITY, NEG_INFINITY}};

use anyhow::{Result, Context};
use cudarc::driver::CudaContext;
use gicp_slam_cuda::{gpu_cov::CudaCovContext, gpu_search::CudaKnnContext, gpu_voxel::CudaVoxelContext, load_files::{load_and_flatten_imu_json, load_pcd_files}, operate_pcd_file::{load_pcd_xyzt, save_pcd_xyz}, pre_process_pcd::{self, preprocess_point_cloud}, predict_pose_imu::{self, build_rotation_trajectory, predict_pose_by_imu}};
use nalgebra::{Matrix3, Matrix6, UnitQuaternion, Vector3, Vector6};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::Solve;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Serialize;


const PCD_DIR: &str = "data/input/mid360/mid360-pointcloud2-bag-outside-station-to-campus/mid360";
const IMU_FILE_PATH: &str = "data/input/mid360/mid360-pointcloud2-bag-outside-station-to-campus/mid360-imu/imu_data.json";
const FINAL_MAP_SAVE_PATH: &str = "data/output/final_map/mid360_gicp_global_map.pcd";

const KNN_PTX_PATH: &str = "src/kernels/search.ptx";
const COV_PTX_PATH: &str = "src/kernels/compute_covariance.ptx";
const VOXEL_PTX_PATH: &str = "src/kernels/voxel.ptx";

const MIN_DIST: f32 = 0.7;
const MAX_DIST: f32 = 35.0;
const VOXEL_SIZE: f32 = 0.5;
const MAX_ITERATIONS: usize = 4;
const LOCAL_MAP_SIZE: usize = 30;
const RMSE_THRESHOLD: f32 = VOXEL_SIZE / 4.0;

const KEYFRAME_DIST_THRESHOLD: f32 = 0.01; // meters
const KEYFRAME_ANGLE_THRESHOLD: f32 = 0.1 * std::f32::consts::PI / 180.0; // radians


#[derive(Serialize)]
struct PoseData {
    timestamp: f64,
    position: [f32; 3],      // x, y, z
    rotation_quat: [f32; 4], // w, x, y, z
}

struct FrameData {
    points: Array2<f32>,
    // covariances: Vec<Matrix3<f64>>,
}

#[derive(Serialize)]
struct RmseStats {
    frame_index: usize,
    rmse: f32,
}

#[derive(Debug, Clone)]
struct TrajectoryRecord {
    min_dist: f32,
    min_degree: f32,
    max_dist: f32,
    max_degree: f32,
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
    total_time: std::time::Duration,
    total_gicp_time: std::time::Duration,
    count: usize,
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
    let gpu_knn = CudaKnnContext::from_context(ctx.clone(), KNN_PTX_PATH)?;
    let mut gpu_cov = CudaCovContext::from_context(ctx.clone(), COV_PTX_PATH)?;
    let mut gpu_voxel = CudaVoxelContext::new(ctx.clone(), VOXEL_PTX_PATH)?;
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

    let mut traj_record = TrajectoryRecord {
        min_dist: INFINITY,
        min_degree: INFINITY,
        max_dist: NEG_INFINITY,
        max_degree: NEG_INFINITY,
    };

    let mut process_time_stats = ProcessTimeStats {
        total_time: std::time::Duration::new(0, 0),
        total_gicp_time: std::time::Duration::new(0, 0),
        count: 0,
    };

    // Initialize global map accumulator
    let mut global_map_accumulator: Vec<Array2<f32>> = Vec::new();
    let init_pcd_arr = pcd_to_array2(&init_pcd);
    global_map_accumulator.push(init_pcd_arr);

    let base_timestamp = imu_data[0].timestamp_sec;
    gicp_odometry.gicp_traj_log.push(
        extract_pose_from_matrix(base_timestamp, &gicp_odometry.current_g_pose)
    );

    let mut last_keyframe_pose = gicp_odometry.current_g_pose.clone();

    for (i, pcd_path) in pcd_paths.iter().enumerate().skip(1) {
        // Load current pcd frame
        let pcd_points = load_pcd_xyzt(pcd_path.to_str().unwrap())
            .expect("Failed to load pcd data");
        
        let total_start = std::time::Instant::now();
        let min_timestamp = pcd_points.iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);

        // Transform the min_timestamp to seconds
        let current_frame_timestamp = min_timestamp / 1_000_000_000.0;

        if i == 1 {
            gicp_odometry.last_timestamp = current_frame_timestamp;
        }

        // Predict pose using IMU data
        let (predicted_pose, predicted_velocity) = predict_pose_by_imu(
            &gicp_odometry.current_g_pose, 
            &gicp_odometry.velocity, 
            gicp_odometry.last_timestamp, 
            current_frame_timestamp, 
            &imu_data
        );

        // let min_timestamp = pcd_points.iter()
        //     .map(|p| p.timestamp)
        //     .fold(f64::INFINITY, f64::min);
        let max_timestamp = pcd_points.iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        // Build rotation trajectory from IMU data
        // let rotation_traj = build_rotation_trajectory(
        //     &imu_data, 
        //     gicp_odometry.last_timestamp, 
        //     current_frame_timestamp
        // );
        let rotation_traj = build_rotation_trajectory(
            &imu_data, 
            min_timestamp, 
            max_timestamp
        );

        // Preprocess point cloud: deskewing and filtering
        let preprocessed_current_points = preprocess_point_cloud(
            &pcd_points, 
            &rotation_traj, 
            MIN_DIST, 
            MAX_DIST
        );

        // let (target_pts, target_covs) = if gicp_odometry.local_map.is_empty() {
        let target_pts = if gicp_odometry.local_map.is_empty() {
            // (preprocessed_current_points.clone(), vec![])
            preprocessed_current_points.clone()
        } else {
            // flatten_local_map(&gicp_odometry.local_map)?
            flatten_local_map(&gicp_odometry.local_map)?
        };

        // Voxel downsample both source and target point clouds
        let voxel_size = VOXEL_SIZE;
        let start = std::time::Instant::now();
        // let v_preprocessed_current_points = voxel_downsample(&preprocessed_current_points, voxel_size);
        let v_preprocessed_current_points = gpu_voxel.voxel_downsample(
            &preprocessed_current_points, 
            preprocessed_current_points.nrows(), 
            voxel_size
        )?;
        let v_target_pts = gpu_voxel.voxel_downsample(
            &target_pts,
            target_pts.nrows(),
            voxel_size
        )?;
        let downsample_duration = start.elapsed();
        println!("Debug: v_source points: {}, v_target points: {}", v_preprocessed_current_points.nrows(), v_target_pts.nrows());
        println!("Voxel downsampling took {:?}", downsample_duration);
        
        // Compute covariances for current frame points
        let computed_source_covs = gpu_cov.compute_covariances(&v_preprocessed_current_points)
            .expect("Failed to compute covariances on GPU");
        let computed_target_covs = gpu_cov.compute_covariances(&v_target_pts)
            .expect("Failed to compute covariances on GPU");

        let mut current_transform = predicted_pose.clone();

        let start = std::time::Instant::now();
        for i in 0..MAX_ITERATIONS {
            // Rotation source points and covariances by predicted pose
            let transformed_source_pts = transform_points(&v_preprocessed_current_points, &current_transform);
            let transformed_source_covs = transform_covariances(
                &computed_source_covs,
                &current_transform
            );

            // let (indices, dists_sq) = gpu_knn.find_nearest(&preprocessed_current_points, &target_pts)
            let (indices, dists_sq) = gpu_knn.find_nearest(&transformed_source_pts, &v_target_pts)
                .expect("Failed to perform GPU k-NN search");

            let max_dist2: f32 = 0.5;

            let mut src_pts = Vec::<f32>::new();
            let mut tgt_pts = Vec::<f32>::new();
            let mut src_covs = Vec::new();
            let mut tgt_covs = Vec::new();

            for j in 0..transformed_source_pts.nrows() {
                let idx = indices[j];
                if idx < 0 {
                    continue;
                }
                if dists_sq[j] > max_dist2 {
                    continue;
                }

                let k = idx as usize;
                if k >= v_target_pts.nrows() {
                    continue;
                }

                src_pts.push(transformed_source_pts[[j, 0]]);
                src_pts.push(transformed_source_pts[[j, 1]]);
                src_pts.push(transformed_source_pts[[j, 2]]);

                tgt_pts.push(v_target_pts[[k, 0]]);
                tgt_pts.push(v_target_pts[[k, 1]]);
                tgt_pts.push(v_target_pts[[k, 2]]);

                src_covs.push(transformed_source_covs[j].clone());
                tgt_covs.push(computed_target_covs[k].clone());
            }

            let n_valid = src_covs.len();
            if n_valid < 20 {
                println!("Too few valid correspondences: {}", n_valid);
                break;
            }

            let src_arr = Array2::from_shape_vec((n_valid, 3), src_pts)
                .expect("Failed to create source Array2");
            let tgt_arr = Array2::from_shape_vec((n_valid, 3), tgt_pts)
                .expect("Failed to create target Array2");

            // GICP
            let delta_t = solve_gicp_step(
                &src_arr, 
                &src_covs, 
                &tgt_arr, 
                &tgt_covs, 
                // &current_transform
            )?;

            current_transform = mat4_mul(&delta_t, &current_transform);

            // Check convergence (RMSE)
            let mut sum = 0.0f32;
            let mut cnt = 0usize;
            for j in 0..transformed_source_pts.nrows() {
                let idx = indices[j];
                if idx < 0 { continue; }
                if dists_sq[j] > max_dist2 { continue; }
                sum += dists_sq[j];
                cnt += 1;
            }
            let rmse = (sum / cnt as f32).sqrt();
            gicp_odometry.rmse_log.push(RmseStats {
                frame_index: i,
                rmse,
            });
            println!("Iteration {}: RMSE = {}", i, rmse);

            if rmse < RMSE_THRESHOLD {
                println!("Converged at iteration {}", i);
                break;
            }
        }
        let gicp_duration = start.elapsed();
        println!("GICP for frame {} took {:?}\n", i, gicp_duration);

        // Update odometry state
        let prev_pose = gicp_odometry.current_g_pose.clone();
        gicp_odometry.current_g_pose = current_transform.clone();

        let dt = current_frame_timestamp - gicp_odometry.last_timestamp;
        if dt > 1e-6 {
            // ndarrayのスライスから直接計算
            let curr_pos = gicp_odometry.current_g_pose.slice(s![0..3, 3]);
            let prev_pos = prev_pose.slice(s![0..3, 3]);
            
            // 速度ベクトル (m/s)
            let vel_x = (curr_pos[0usize] - prev_pos[0usize]) / dt as f32;
            let vel_y = (curr_pos[1usize] - prev_pos[1usize]) / dt as f32;
            let vel_z = (curr_pos[2usize] - prev_pos[2usize]) / dt as f32;

            // Max velocity is 2.0 m/s
            gicp_odometry.velocity = Vector3::new(vel_x as f64, vel_y as f64, vel_z as f64)
                .cap_magnitude(2.0); 
        }

        gicp_odometry.last_timestamp = current_frame_timestamp;

        // 並進距離の計算
        let current_pos = gicp_odometry.current_g_pose.slice(s![0..3, 3]);
        let last_key_pos = last_keyframe_pose.slice(s![0..3, 3]);
        let delta_dist = ((current_pos[0usize] - last_key_pos[0usize]).powi(2) + 
                          (current_pos[1usize] - last_key_pos[1usize]).powi(2) + 
                          (current_pos[2usize] - last_key_pos[2usize]).powi(2)).sqrt();

        // 回転角の計算 (トレースから概算)
        // R_delta = R_last^T * R_curr
        // Angle = arccos((tr(R_delta) - 1) / 2)
        let r_curr = gicp_odometry.current_g_pose.slice(s![0..3, 0..3]);
        let r_last = last_keyframe_pose.slice(s![0..3, 0..3]);
        // R_delta のトレース計算 (行列積の対角成分の和)
        let tr_r_delta = 
            (r_last[[0,0]]*r_curr[[0,0]] + r_last[[1,0]]*r_curr[[1,0]] + r_last[[2,0]]*r_curr[[2,0]]) + // (R_last^T)_row0 * R_curr_col0
            (r_last[[0,1]]*r_curr[[0,1]] + r_last[[1,1]]*r_curr[[1,1]] + r_last[[2,1]]*r_curr[[2,1]]) + 
            (r_last[[0,2]]*r_curr[[0,2]] + r_last[[1,2]]*r_curr[[1,2]] + r_last[[2,2]]*r_curr[[2,2]]);
        
        let delta_angle = ((tr_r_delta - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

        // 2. 閾値チェック (初回の数フレームは強制的に追加しても良い)
        let is_keyframe = delta_dist > KEYFRAME_DIST_THRESHOLD || delta_angle > KEYFRAME_ANGLE_THRESHOLD;

        // if is_keyframe {
        if i % 3 == 0 {
            // let delta_degree = delta_angle * 180.0 / std::f32::consts::PI;
            // println!("Frame {} is a keyframe (Δdist: {:.3} m, Δangle: {:.3} deg)", 
            //     i, delta_dist, delta_degree);
            println!("Frame {} is a keyframe (Δdist: {:.3} m, Δangle: {:.3} rad)", 
                i, delta_dist, delta_angle);

            if delta_dist < traj_record.min_dist {
                // println!("  Updating min_dist: {:.3} -> {:.3}", traj_record.min_dist, delta_dist);
                traj_record.min_dist = delta_dist;
            }
            if delta_dist > traj_record.max_dist {
                // println!("  Updating max_dist: {:.3} -> {:.3}", traj_record.max_dist, delta_dist);
                traj_record.max_dist = delta_dist;
            }
            if delta_angle < traj_record.min_degree {
                // println!("  Updating min_angle: {:.3} -> {:.3}", traj_record.min_angle, delta_angle);
                traj_record.min_degree = delta_angle;
            }
            if delta_angle > traj_record.max_degree {
                // println!("  Updating max_angle: {:.3} -> {:.3}", traj_record.max_angle, delta_angle);
                traj_record.max_degree = delta_angle;
            }

            last_keyframe_pose = gicp_odometry.current_g_pose.clone();
            // Rotation current points to global frame
            let aligned_current_points = transform_points(&preprocessed_current_points, &gicp_odometry.current_g_pose);

            // Update local and global maps
            if gicp_odometry.local_map.len() >= LOCAL_MAP_SIZE {
                gicp_odometry.local_map.pop_front();
            }

            gicp_odometry.local_map.push_back(FrameData {
                points: aligned_current_points.clone(),
            });

            if (i % 6) == 0 {
                global_map_accumulator.push(aligned_current_points.clone());
            }
        }

        // Update trajectory log
        gicp_odometry.gicp_traj_log.push(
            extract_pose_from_matrix(current_frame_timestamp, &gicp_odometry.current_g_pose)
        );
        let total_duration = total_start.elapsed();
        process_time_stats.total_time += total_duration;
        process_time_stats.total_gicp_time += gicp_duration;
        process_time_stats.count += 1;
        
    }

    println!();
    println!("Trajectory Record:");
    println!("  Min Distance: {:.4} m", traj_record.min_dist);
    println!("  Max Distance: {:.4} m", traj_record.max_dist);
    println!("  Min Degree   : {:.4} deg", traj_record.min_degree  * 180.0 / std::f32::consts::PI);
    println!("  Max Degree   : {:.4} deg", traj_record.max_degree * 180.0 / std::f32::consts::PI);
    println!();
    print_parameters(&process_time_stats);

    let final_global_map = ndarray::concatenate(
        Axis(0), 
    &global_map_accumulator.iter().map(|arr| arr.view()).collect::<Vec<_>>()
    ).context("Failed to concatenate global map")?;

    let voxel_size = 0.1;
    let v_final_global_map = voxel_downsample(&final_global_map, voxel_size);
    let final_pcd = array2_to_pcd(&v_final_global_map);
    save_pcd_xyz(&final_pcd, FINAL_MAP_SAVE_PATH)
        .context("Failed to save final global map PCD")?;
    Ok(())
}

fn print_parameters(process_time_stats: &ProcessTimeStats) {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║          GICP-SLAM-CUDA Configuration Parameters         ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Input/Output Paths:                                       ║");
    println!("║   PCD Directory      : {:<35} ║", PCD_DIR);
    println!("║   IMU File           : {:<35} ║", IMU_FILE_PATH);
    println!("║   Output Map         : {:<35} ║", FINAL_MAP_SAVE_PATH);
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ CUDA Kernel Paths:                                        ║");
    println!("║   KNN PTX            : {:<35} ║", KNN_PTX_PATH);
    println!("║   Covariance PTX     : {:<35} ║", COV_PTX_PATH);
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Point Cloud Processing:                                   ║");
    println!("║   Min Distance       : {:<7.2} m                            ║", MIN_DIST);
    println!("║   Max Distance       : {:<7.2} m                            ║", MAX_DIST);
    println!("║   Voxel Size         : {:<7.2} m                            ║", VOXEL_SIZE);
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ GICP Optimization:                                        ║");
    println!("║   Max Iterations     : {:<7}                               ║", MAX_ITERATIONS);
    println!("║   RMSE Threshold     : {:<7.4} m                            ║", RMSE_THRESHOLD);
    println!("║   Correspondence Dist: {:<7.2} m                            ║", 0.5_f32.sqrt());
    println!("║   Local Map Size     : {:<7} frames                        ║", LOCAL_MAP_SIZE);
    println!("║   Max Velocity       : {:<7.2} m/s                          ║", 2.0);
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Map Accumulation:                                         ║");
    println!("║   Global Map Stride  : Every 3 frames                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                   Processing Summary                     ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    // println!("║ Frames Processed     : {:<7}                               ║", process_time_stats.count);
    // println!("║ Total Time           : {:<10.2?}                          ║", process_time_stats.total_time);
    // println!("║ Total GICP Time      : {:<10.2?}                          ║", process_time_stats.total_gicp_time);
    // println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Average Per Frame:                                        ║");
    println!("║   Total              : {:<10.2?}                          ║", 
        process_time_stats.total_time / (process_time_stats.count as u32));
    println!("║   GICP Only          : {:<10.2?}                          ║", 
        process_time_stats.total_gicp_time / (process_time_stats.count as u32));
    println!("╠═══════════════════════════════════════════════════════════╣");
    // println!("║ Output:                                                   ║");
    // println!("║   Global Map Points  : {:<7}                               ║", v_final_global_map.nrows());
    // println!("║   Trajectory Poses   : {:<7}                               ║", gicp_odometry.gicp_traj_log.len());
    // println!("║   RMSE Log Entries   : {:<7}                               ║", gicp_odometry.rmse_log.len());
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Files Saved:                                              ║");
    println!("║   Global Map         : {:<35} ║", FINAL_MAP_SAVE_PATH);
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
}

fn pcd_to_array2(pcd_points: &[gicp_slam_cuda::operate_pcd_file::PointXYZT]) -> Array2<f32> {
    let n = pcd_points.len();
    let mut arr = Array2::<f32>::zeros((n, 3));
    
    for (i, pt) in pcd_points.iter().enumerate() {
        arr[[i, 0]] = pt.x;
        arr[[i, 1]] = pt.y;
        arr[[i, 2]] = pt.z;
    }
    
    arr
}

fn array2_to_pcd(arr: &Array2<f32>) -> Vec<gicp_slam_cuda::operate_pcd_file::PointXYZ> {
    let n = arr.nrows();
    let mut pcd_points = Vec::with_capacity(n);
    
    for i in 0..n {
        pcd_points.push(gicp_slam_cuda::operate_pcd_file::PointXYZ {
            x: arr[[i, 0]],
            y: arr[[i, 1]],
            z: arr[[i, 2]],
        });
    }
    
    pcd_points
}

fn mat4_mul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    // a(4x4) * b(4x4)
    let mut out = Array2::<f32>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0f32;
            for k in 0..4 {
                s += a[[i,k]] * b[[k,j]];
            }
            out[[i,j]] = s;
        }
    }
    out
}

fn solve_gicp_step(
    source_pts: &Array2<f32>,           // 現在位置にあるソース点 (N x 3)
    source_covs: &[Matrix3<f64>],       // ソースの共分散 (初期姿勢での計算値)
    target_pts: &Array2<f32>,           // 対応するターゲット点 (N x 3)
    target_covs: &[Matrix3<f64>],       // 対応するターゲットの共分散
    // current_transform: &Array2<f32>,    // 現在の推定変換行列 (4x4)
) -> Result<Array2<f32>> { // 戻り値: 微小移動行列 Delta T

    let n = source_pts.nrows();
    
    // 現在の回転行列 R を抽出 (ソースの共分散を回転させるため)
    // let r_curr = Matrix3::new(
    //     current_transform[[0,0]] as f64, current_transform[[0,1]] as f64, current_transform[[0,2]] as f64,
    //     current_transform[[1,0]] as f64, current_transform[[1,1]] as f64, current_transform[[1,2]] as f64,
    //     current_transform[[2,0]] as f64, current_transform[[2,1]] as f64, current_transform[[2,2]] as f64,
    // );

    // Rayonで並列化して H と b を計算し、最後にsumする
    // let (h_sum, b_sum) = (0..n).into_par_iter()
    // ループの外でアロケーションなしで初期化（スタック領域）
    let mut h_sum = Matrix6::<f64>::zeros();
    let mut b_sum = Vector6::<f64>::zeros();

    // イテレータではなく単純なforループの方がコンパイラ最適化がかかりやすい場合がある
    for i in 0..n {
        let p_s = Vector3::new(source_pts[[i,0]] as f64, source_pts[[i,1]] as f64, source_pts[[i,2]] as f64);
        let p_t = Vector3::new(target_pts[[i,0]] as f64, target_pts[[i,1]] as f64, target_pts[[i,2]] as f64);
        
        // 1. マハラノビス距離の重み行列
        let c_sum = target_covs[i] + source_covs[i];
        
        let omega = match c_sum.try_inverse() {
            Some(inv) => inv,
            None => continue, // 逆行列なしならスキップ
        };

        // 2. 誤差ベクトル
        let error = p_t - p_s;

        // 3. J^T * Omega * error (bの一部)
        let w_e = omega * error;
        
        // Rotational part of b: p_s x (Omega * error)
        let cross = p_s.cross(&w_e);
        
        // b_sum に直接加算 (配列生成コストなし)
        b_sum[0] += cross.x;
        b_sum[1] += cross.y;
        b_sum[2] += cross.z;
        b_sum[3] += w_e.x;
        b_sum[4] += w_e.y;
        b_sum[5] += w_e.z;

        // 4. H行列の構築
        // Omega * Skew(p_s) の計算
        // S_col0 = [0, -z, y]^T なので Omega * S_col0 を計算
        let x = p_s.x; let y = p_s.y; let z = p_s.z;
        
        let w_s0 = omega * Vector3::new(0.0, -z, y);
        let w_s1 = omega * Vector3::new(z, 0.0, -x);
        let w_s2 = omega * Vector3::new(-y, x, 0.0);

        // 左上: Skew(p_s)^T * Omega * Skew(p_s) = p_s x (Omega * Skew_col)
        let h00 = p_s.cross(&w_s0);
        let h01 = p_s.cross(&w_s1);
        let h02 = p_s.cross(&w_s2);

        // h_sum に直接加算
        // 左上 (3x3)
        h_sum[(0,0)] += h00.x; h_sum[(0,1)] += h01.x; h_sum[(0,2)] += h02.x;
        h_sum[(1,0)] += h00.y; h_sum[(1,1)] += h01.y; h_sum[(1,2)] += h02.y;
        h_sum[(2,0)] += h00.z; h_sum[(2,1)] += h01.z; h_sum[(2,2)] += h02.z;

        // 右下 (3x3) = Omega
        h_sum[(3,3)] += omega[(0,0)]; h_sum[(3,4)] += omega[(0,1)]; h_sum[(3,5)] += omega[(0,2)];
        h_sum[(4,3)] += omega[(1,0)]; h_sum[(4,4)] += omega[(1,1)]; h_sum[(4,5)] += omega[(1,2)];
        h_sum[(5,3)] += omega[(2,0)]; h_sum[(5,4)] += omega[(2,1)]; h_sum[(5,5)] += omega[(2,2)];

        // 右上 (3x3) = [w_s0, w_s1, w_s2]^T
        h_sum[(0,3)] += w_s0.x; h_sum[(0,4)] += w_s0.y; h_sum[(0,5)] += w_s0.z;
        h_sum[(1,3)] += w_s1.x; h_sum[(1,4)] += w_s1.y; h_sum[(1,5)] += w_s1.z;
        h_sum[(2,3)] += w_s2.x; h_sum[(2,4)] += w_s2.y; h_sum[(2,5)] += w_s2.z;

        // 左下 (3x3) = 右上の転置
        h_sum[(3,0)] += w_s0.x; h_sum[(3,1)] += w_s1.x; h_sum[(3,2)] += w_s2.x;
        h_sum[(4,0)] += w_s0.y; h_sum[(4,1)] += w_s1.y; h_sum[(4,2)] += w_s2.y;
        h_sum[(5,0)] += w_s0.z; h_sum[(5,1)] += w_s1.z; h_sum[(5,2)] += w_s2.z;
    };

    let h_arr = Array2::from_shape_fn((6, 6), |(r, c)| h_sum[(r, c)]);
    let b_arr = Array1::from_shape_fn(6, |i| b_sum[i]);
    
    // H x = b を解く
    // ここは前のコードと同じ (solve or SVD fallback)
    let delta = solve_linear_system_6x6(h_arr, b_arr)?;
    
    Ok(delta)
}

fn solve_linear_system_6x6(a: Array2<f64>, b: Array1<f64>) -> Result<Array2<f32>> {
    let x = a.solve(&b).or_else(|_| {
         Err(anyhow::anyhow!("Linear solve failed"))
    })?;

    // x = [alpha, beta, gamma, tx, ty, tz]
    let delta_matrix = convert_se3_to_matrix4(x);
    Ok(delta_matrix)
}

// [alpha, beta, gamma, tx, ty, tz] -> 4x4 matrix
fn convert_se3_to_matrix4(x: Array1<f64>) -> Array2<f32> {
    let alpha = x[0]; let beta = x[1]; let gamma = x[2];
    let tx = x[3]; let ty = x[4]; let tz = x[5];

    let theta = (alpha*alpha + beta*beta + gamma*gamma).sqrt();
    let r: Array2<f64>;

    if theta < 1e-9 {
        r = ndarray::array![
            [1.0, -gamma, beta],
            [gamma, 1.0, -alpha],
            [-beta, alpha, 1.0]
        ];
    } else {
        let k_x = alpha / theta;
        let k_y = beta / theta;
        let k_z = gamma / theta;
        let c = theta.cos();
        let s = theta.sin();
        let v = 1.0 - c;

        r = ndarray::array![
            [k_x*k_x*v + c,     k_x*k_y*v - k_z*s, k_x*k_z*v + k_y*s],
            [k_x*k_y*v + k_z*s, k_y*k_y*v + c,     k_y*k_z*v - k_x*s],
            [k_x*k_z*v - k_y*s, k_y*k_z*v + k_x*s, k_z*k_z*v + c]
        ];
    }

    ndarray::array![
        [r[[0,0]] as f32, r[[0,1]] as f32, r[[0,2]] as f32, tx as f32],
        [r[[1,0]] as f32, r[[1,1]] as f32, r[[1,2]] as f32, ty as f32],
        [r[[2,0]] as f32, r[[2,1]] as f32, r[[2,2]] as f32, tz as f32],
        [0.0, 0.0, 0.0, 1.0]
    ]
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
) -> Result<Array2<f32>> {
    
    // 1. 点群 (Array2) の結合
    let points_views: Vec<_> = queue.iter()
        .map(|frame| frame.points.view())
        .collect();

    // Axis(0) = 行方向（縦）に結合
    let merged_points = ndarray::concatenate(Axis(0), &points_views)
        .context("Failed to concatenate local map points")?;

    // 2. 共分散 (Vec) の結合
    // let total_points = merged_points.nrows();
    // let mut merged_covs = Vec::with_capacity(total_points);

    // for frame in queue {
    //     merged_covs.extend_from_slice(&frame.covariances);
    // }

    // 整合性チェック (念のため)
    // if merged_points.nrows() != merged_covs.len() {
    //     return Err(anyhow::anyhow!(
    //         "Mismatch between points count ({}) and covariances count ({}) in local map",
    //         merged_points.nrows(),
    //         merged_covs.len()
    //     ));
    // }

    // Ok((merged_points, merged_covs))
    Ok(merged_points)
}

fn voxel_downsample(
    pts: &Array2<f32>,
    voxel_size: f32
) -> Array2<f32> {
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

        grid.entry(key)
            // 既にボクセルに点がある場合：座標を足し合わせ、カウントを増やす
            .and_modify(|(sum, count)| {
                *sum += p_vec;
                *count += 1;
            })
            // 初めての点の場合：座標、カウント1、そして共分散を保存
            .or_insert((p_vec, 1));
    }

    // 抽出（重心を計算）
    let n_kept = grid.len();
    let mut new_pts = Array2::<f32>::zeros((n_kept, 3));

    for (k, (_, (sum, count))) in grid.iter().enumerate() {
        // 重心 = 合計 / 個数
        let centroid = sum / (*count as f32);
        
        new_pts[[k, 0]] = centroid.x;
        new_pts[[k, 1]] = centroid.y;
        new_pts[[k, 2]] = centroid.z;    
    }

    new_pts
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

fn transform_covariances(
    covs: &[Matrix3<f64>],
    transform: &Array2<f32>,
) -> Vec<Matrix3<f64>> {
    // Extract 3×3 rotation matrix from 4×4 transform
    let r = Matrix3::new(
        transform[[0, 0]] as f64, transform[[0, 1]] as f64, transform[[0, 2]] as f64,
        transform[[1, 0]] as f64, transform[[1, 1]] as f64, transform[[1, 2]] as f64,
        transform[[2, 0]] as f64, transform[[2, 1]] as f64, transform[[2, 2]] as f64,
    );
    
    // C' = R * C * R^T for each covariance
    covs.iter()
        .map(|cov| r * cov * r.transpose())
        .collect()
}