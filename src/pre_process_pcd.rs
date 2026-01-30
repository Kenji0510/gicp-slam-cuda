use nalgebra::{UnitQuaternion, Vector3};
use ndarray::Array2;

use crate::{operate_pcd_file::PointXYZT, predict_pose_imu::RotationTrajectory};


pub fn preprocess_point_cloud(
    points: &[PointXYZT],
    trajectory: &RotationTrajectory,
    min_dist: f32,
    max_dist: f32,
) -> Array2<f32> {
    let n_points = points.len();
    let mut valid_points_flat = Vec::with_capacity(n_points * 3);

    // 1. このフレームの基準となる回転を取得（通常は先頭の点の時刻）
    let frame_start_time = points[0].timestamp / 1_000_000_000.0; // ナノ秒→秒変換
    let start_rotation = get_rotation_at_time(trajectory, frame_start_time);
    
    // 基準回転の逆行列を事前に計算（R_start^-1）
    let start_rotation_inv = start_rotation.inverse();

    for p in points {
        let x = p.x as f32;
        let y = p.y as f32;
        let z = p.z as f32;

        // 距離フィルタ
        let dist_sq = x * x + y * y + z * z;
        if dist_sq < min_dist * min_dist || dist_sq > max_dist * max_dist {
            continue;
        }

        // 2. その点の時刻の回転を取得 (R_current)
        let point_time = p.timestamp; // 必要に応じて time_offset 加算
        let current_rotation = get_rotation_at_time(trajectory, point_time);

        // 3. 相対回転 (Relative Rotation) を計算
        // R_relative = R_start^-1 * R_current
        // これにより、フレーム先頭時刻からその点までの「差分回転」が得られます
        let relative_rotation = start_rotation_inv * current_rotation;

        // 4. 座標変換
        let p_vec = Vector3::new(x as f64, y as f64, z as f64);
        
        // ★修正: inverse()せよ、ではなく「相対回転」をそのまま適用
        // センサーが回転した分だけ、点を同じ方向に回して戻してあげるイメージ
        let corrected = relative_rotation * p_vec;

        if corrected.x.is_nan() || corrected.y.is_nan() || corrected.z.is_nan() {
            continue;
        }

        valid_points_flat.push(corrected.x as f32);
        valid_points_flat.push(corrected.y as f32);
        valid_points_flat.push(corrected.z as f32);
    }

    let n_valid = valid_points_flat.len() / 3;
    Array2::from_shape_vec((n_valid, 3), valid_points_flat)
        .expect("Failed to create Array2 from valid points")
}

fn get_rotation_at_time(traj: &RotationTrajectory, t: f64) -> UnitQuaternion<f64> {
    if traj.is_empty() { return UnitQuaternion::identity(); }
    if t <= traj.first().unwrap().0 { return traj.first().unwrap().1; }
    if t >= traj.last().unwrap().0 { return traj.last().unwrap().1; }

    // 線形探索
    for i in 0..traj.len()-1 {
        let (t0, q0) = traj[i];
        let (t1, q1) = traj[i+1];
        
        if t >= t0 && t <= t1 {
            let denom = t1 - t0;
            if denom.abs() < 1e-9 {
                return q0;
            }

            let ratio = (t - t0) / denom;
            
            return q0.slerp(&q1, ratio);
        }
    }
    traj.last().unwrap().1
}