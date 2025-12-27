use nalgebra::{Matrix3, Unit, UnitQuaternion, Vector3};
use ndarray::Array2;

use crate::load_files::ImuSample;


pub fn predict_pose_by_imu(
    start_pose_mat: &Array2<f32>, // 前回のGICP収束後の姿勢 (4x4)
    start_vel: &Vector3<f64>,     // 前回の速度
    start_time: f64,              // 前回のタイムスタンプ
    end_time: f64,                // 今回のタイムスタンプ
    imu_samples: &[ImuSample],    // 全IMUデータ
) -> (Array2<f32>, Vector3<f64>) { // (予測姿勢, 予測速度)
    
    // 1. Array2<f32> から nalgebra の型 (Isometry3/UnitQuaternion) に変換
    let tx = start_pose_mat[[0, 3]] as f64;
    let ty = start_pose_mat[[1, 3]] as f64;
    let tz = start_pose_mat[[2, 3]] as f64;
    let mut position = Vector3::new(tx, ty, tz);

    let mat3 = Matrix3::new(
        start_pose_mat[[0, 0]] as f64, start_pose_mat[[0, 1]] as f64, start_pose_mat[[0, 2]] as f64,
        start_pose_mat[[1, 0]] as f64, start_pose_mat[[1, 1]] as f64, start_pose_mat[[1, 2]] as f64,
        start_pose_mat[[2, 0]] as f64, start_pose_mat[[2, 1]] as f64, start_pose_mat[[2, 2]] as f64,
    );
    let mut rotation = UnitQuaternion::from_matrix(&mat3);
    let mut velocity = *start_vel;

    // 重力ベクトル (World frame, Z-upと仮定)
    let gravity = Vector3::new(0.0, 0.0, 9.80665);

    // 2. 指定範囲のIMUデータを抽出
    // 前回の終わりから今回の終わりまでを含めるため少しバッファを持たせるか、厳密にフィルタリングする
    let relevant_samples: Vec<&ImuSample> = imu_samples.iter()
        .filter(|s| s.timestamp_sec > start_time && s.timestamp_sec <= end_time)
        .collect();

    let mut last_t = start_time;

    // 3. 積分 (Dead Reckoning)
    for sample in relevant_samples {
        let dt = sample.timestamp_sec - last_t;
        if dt <= 1e-9 { continue; }

        // --- 回転の更新 (Gyro) ---
        let wx = sample.gyro[0] as f64;
        let wy = sample.gyro[1] as f64;
        let wz = sample.gyro[2] as f64;
        let omega = Vector3::new(wx, wy, wz);
        
        let angle = omega.norm() * dt;
        let axis = if angle < 1e-9 { Vector3::x_axis() } else { Unit::new_normalize(omega) };
        let delta_q = UnitQuaternion::from_axis_angle(&axis, angle);
        
        rotation = rotation * delta_q; // Global frame orientation update
        rotation.renormalize();

        // --- 速度・位置の更新 (Accel) ---
        let ax = sample.linear_acceleration[0] as f64;
        let ay = sample.linear_acceleration[1] as f64;
        let az = sample.linear_acceleration[2] as f64;
        let acc_local = Vector3::new(ax, ay, az);

        // ローカル加速度をグローバルへ変換
        let acc_global = rotation * acc_local;
        
        // 重力除去
        let acc_net = acc_global - gravity;

        // 等加速度運動として積分
        position += velocity * dt + 0.5 * acc_net * dt * dt;
        velocity += acc_net * dt;

        last_t = sample.timestamp_sec;
    }

    // 4. nalgebra -> Array2<f32> (4x4 Matrix) に戻す
    let r_mat = rotation.to_rotation_matrix();
    let r = r_mat.matrix();
    
    let predicted_mat = ndarray::array![
        [r[(0,0)] as f32, r[(0,1)] as f32, r[(0,2)] as f32, position.x as f32],
        [r[(1,0)] as f32, r[(1,1)] as f32, r[(1,2)] as f32, position.y as f32],
        [r[(2,0)] as f32, r[(2,1)] as f32, r[(2,2)] as f32, position.z as f32],
        [0.0,             0.0,             0.0,             1.0]
    ];

    (predicted_mat, velocity)
}

pub type RotationTrajectory = Vec<(f64, UnitQuaternion<f64>)>;

pub fn build_rotation_trajectory(
    imu_samples: &[ImuSample], 
    start_time: f64,
    end_time: f64,
) -> RotationTrajectory {
    let mut trajectory = Vec::new();
    let mut current_rotation = UnitQuaternion::identity();
    
    // 範囲内のデータのみ抽出
    let buffer_time = 0.01; // 10ms余裕を持たせる
    let search_start = start_time - buffer_time;
    let search_end = end_time + buffer_time;

    let relevant_samples: Vec<&ImuSample> = imu_samples.iter()
        .filter(|s| s.timestamp_sec >= search_start && s.timestamp_sec <= search_end)
        .collect();

    // 最初の基準点
    trajectory.push((search_start, current_rotation));

    let mut last_time = search_start;

    for sample in relevant_samples {
        let dt = sample.timestamp_sec - last_time;

        if dt <= 1e-9 { 
            continue; 
        }

        let wx = sample.gyro[0] as f64;
        let wy = sample.gyro[1] as f64;
        let wz = sample.gyro[2] as f64;
        let omega = Vector3::new(wx, wy, wz);

        // 微小回転を今の回転に積み上げる
        let angle_axis = omega * dt;
        let delta_q = UnitQuaternion::new(angle_axis);
        current_rotation = current_rotation * delta_q;

        // ★修正: 誤差蓄積を防ぐため正規化する
        current_rotation.renormalize();

        trajectory.push((sample.timestamp_sec, current_rotation));
        last_time = sample.timestamp_sec;
    }
    
    trajectory
}