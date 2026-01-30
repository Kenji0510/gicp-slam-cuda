use std::{fs::{self, File}, io::BufReader, path::PathBuf};

use anyhow::{Context, Result};
use ndarray::{Array1, arr1};
use serde::Deserialize;


#[derive(Debug, Clone)]
pub struct ImuSample {
    pub timestamp_sec: f64,
    pub gyro: Array1<f32>,
    pub linear_acceleration: Array1<f32>,
}

#[derive(Debug, Deserialize)]
pub struct LivoxImu {
    pub timestamp: u64, // Nanoseconds
    pub angular_velocity: [f32; 3],
    pub linear_acceleration: [f32; 3],
}

pub fn load_pcd_files(
    dir_path: &str,
) -> Result<Vec<PathBuf>> {
    // let re = regex::Regex::new(r"voxelized-005_frame_(\d+)\.pcd$")
    let re = regex::Regex::new(r"frame_(\d+)\.pcd$")
        .context("Invalid regex pattern")?;

    let entries = fs::read_dir(dir_path)
        .context(format!("Failed to read directory: {}", dir_path))?;

    let mut files_with_numbers: Vec<(PathBuf, u32)> = Vec::new();

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => continue,
        };

        if let Some(captures) = re.captures(filename) {
            if let Some(num_str) = captures.get(1) {
                if let Ok(num) = num_str.as_str().parse::<u32>() {
                    files_with_numbers.push((path.clone(), num));
                }
            }
        }
    }

    files_with_numbers.sort_by_key(|(_path, num)| *num);

    let sorted_paths: Vec<PathBuf> = files_with_numbers
        .into_iter()
        .map(|(path, _num)| path)
        .collect();

    Ok(sorted_paths)
}

pub fn load_and_flatten_imu_json(path: &str) -> Result<Vec<ImuSample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let batches: Vec<LivoxImu> = serde_json::from_reader(reader)?;

    let mut samples = Vec::new();

    for batch in batches {
        samples.push(ImuSample {
            timestamp_sec: batch.timestamp as f64 / 1_000_000_000.0,
            gyro: arr1(&[
                batch.angular_velocity[0] as f32,
                batch.angular_velocity[1] as f32,
                batch.angular_velocity[2] as f32,
            ]),
            linear_acceleration: arr1(&[
                batch.linear_acceleration[0] as f32,
                batch.linear_acceleration[1] as f32,
                batch.linear_acceleration[2] as f32,
            ]),
        });
    }

    samples.sort_by(|a, b| a.timestamp_sec.partial_cmp(&b.timestamp_sec).unwrap());

    Ok(samples)
}