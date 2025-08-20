//! Basic EBCC compression example.
//! 
//! This example demonstrates how to use the EBCC Rust bindings for
//! compressing and decompressing climate data.

use ebcc::{encode_climate_variable, decode_climate_variable, EBCCConfig, init_logging};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    init_logging();
    
    println!("EBCC Basic Compression Example");
    println!("=============================");
    
    // Create some synthetic climate data (ERA5-like grid)
    let height = 721;
    let width = 1440;
    let frames = 1;
    let total_elements = frames * height * width;
    
    // Generate synthetic temperature data (in Kelvin)
    let mut data = Vec::with_capacity(total_elements);
    for i in 0..height {
        for j in 0..width {
            // Simple synthetic temperature field with spatial variation
            let lat = -90.0 + (i as f32 / height as f32) * 180.0;
            let lon = -180.0 + (j as f32 / width as f32) * 360.0;
            
            // Temperature decreases with latitude, with some variation
            let temp = 273.15 + 30.0 * (1.0 - lat.abs() / 90.0) 
                + 5.0 * (lon / 180.0).sin() 
                + 2.0 * (lat / 90.0 * 4.0).sin();
            
            data.push(temp);
        }
    }
    
    println!("Generated {} climate data points", total_elements);
    println!("Data range: {:.2} to {:.2} K", 
             data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Test different compression configurations
    let configs = vec![
        ("JPEG2000 only (CR=10)", EBCCConfig::jpeg2000_only([frames, height, width], 10.0)),
        ("JPEG2000 only (CR=30)", EBCCConfig::jpeg2000_only([frames, height, width], 30.0)),
        ("Max error bound (0.1K)", EBCCConfig::max_error_bounded([frames, height, width], 20.0, 0.1)),
        ("Max error bound (0.01K)", EBCCConfig::max_error_bounded([frames, height, width], 20.0, 0.01)),
        ("Relative error (0.1%)", EBCCConfig::relative_error_bounded([frames, height, width], 20.0, 0.001)),
    ];
    
    let original_size = total_elements * std::mem::size_of::<f32>();
    
    for (name, config) in configs {
        println!("\n--- {} ---", name);
        
        // Compress the data
        let start = std::time::Instant::now();
        let compressed = encode_climate_variable(&data, &config)?;
        let compress_time = start.elapsed();
        
        // Decompress the data
        let start = std::time::Instant::now();
        let decompressed = decode_climate_variable(&compressed)?;
        let decompress_time = start.elapsed();
        
        // Calculate compression metrics
        let compression_ratio = original_size as f64 / compressed.len() as f64;
        let compressed_size_mb = compressed.len() as f64 / (1024.0 * 1024.0);
        let original_size_mb = original_size as f64 / (1024.0 * 1024.0);
        
        // Calculate error metrics
        let max_error = data.iter().zip(decompressed.iter())
            .map(|(&orig, &decomp)| (orig - decomp).abs())
            .fold(0.0f32, f32::max);
        
        let mse: f64 = data.iter().zip(decompressed.iter())
            .map(|(&orig, &decomp)| ((orig - decomp) as f64).powi(2))
            .sum::<f64>() / total_elements as f64;
        let rmse = mse.sqrt();
        
        // Calculate relative error
        let data_range = data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
            .max(data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
            - data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_relative_error = max_error / data_range * 100.0;
        
        println!("  Original size:     {:.2} MB", original_size_mb);
        println!("  Compressed size:   {:.2} MB", compressed_size_mb);
        println!("  Compression ratio: {:.2}:1", compression_ratio);
        println!("  Compression time:  {:.2} ms", compress_time.as_secs_f64() * 1000.0);
        println!("  Decompression time: {:.2} ms", decompress_time.as_secs_f64() * 1000.0);
        println!("  Max error:         {:.4} K", max_error);
        println!("  RMSE:              {:.4} K", rmse);
        println!("  Max relative error: {:.4}%", max_relative_error);
    }
    
    println!("\nCompression example completed successfully!");
    Ok(())
}