use pcd_rs::{PcdDeserialize, PcdSerialize, Reader};
use anyhow::Result;


#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZT {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub timestamp: f64,
}

pub fn save_pcd_xyzt(
    points: &[PointXYZT],
    file_path: &str,
) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;
    
    for point in points {
        writer.push(point)?;
    }
    writer.finish()?;

    Ok(())
}

pub fn load_pcd_xyzt(
    file_path: &str,
) -> Result<Vec<PointXYZT>> {
    let reader = match Reader::open(file_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open PCD file: {}", e);
            return Err(anyhow::anyhow!("Failed to open PCD file: {}", e));
        }
    };

    let points: Vec<PointXYZT> = match reader.collect() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to read PCD data: {}", e);
            return Err(anyhow::anyhow!("Failed to read PCD data: {}", e));
        }
    };

    Ok(points)
}