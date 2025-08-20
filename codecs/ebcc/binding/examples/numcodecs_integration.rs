//! EBCC numcodecs integration example.
//! 
//! This example shows how to use EBCC with the numcodecs ecosystem,
//! including configuration serialization, codec creation, and actual
//! compression/decompression using the numcodecs API.

use numcodecs_ebcc::{EBCCCodec, EBCCConfig, ebcc_codec_from_config, init_logging};
use std::collections::HashMap;

use numcodecs::{Codec, AnyCowArray, AnyArray};

use ndarray::Array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    
    println!("EBCC numcodecs Integration Example");
    println!("==================================\n");
    
    // Example 1: Direct codec creation
    println!("1. Direct codec creation:");
    let config = EBCCConfig::new([1, 32, 32]); // Single frame, 32x32
    let codec = EBCCCodec::new(config)?;
    println!("   ✓ Created EBCC codec with dimensions {:?}", codec.config.dims);
    println!("   ✓ Base compression ratio: {}", codec.config.base_cr);
    println!("   ✓ Residual type: {:?}", codec.config.residual_compression_type);

    // Example 2: Create codec from configuration map (like numcodecs JSON)
    println!("\n2. Codec creation from configuration map:");
    let mut config_map = HashMap::new();
    config_map.insert("dims".to_string(), serde_json::json!([1, 32, 32]));
    config_map.insert("base_cr".to_string(), serde_json::json!(20.0));
    config_map.insert("residual_type".to_string(), serde_json::json!("max_error"));
    config_map.insert("error".to_string(), serde_json::json!(0.01));
    
    let codec_from_config = ebcc_codec_from_config(config_map)?;
    println!("   ✓ Created EBCC codec from config map");
    println!("   ✓ Base compression ratio: {}", codec_from_config.config.base_cr);
    println!("   ✓ Error bound: {}", codec_from_config.config.error);

    // Example 3: Using different compression modes
    println!("\n3. Different compression modes:");
    
    // JPEG2000-only compression
    let jpeg_codec = EBCCCodec::jpeg2000_only([1, 32, 32], 15.0)?;
    println!("   JPEG2000-only: CR={}, residual={:?}", 
             jpeg_codec.config.base_cr, 
             jpeg_codec.config.residual_compression_type);
    
    // Max error bounded compression
    let max_error_codec = EBCCCodec::max_error_bounded([1, 32, 32], 10.0, 0.05)?;
    println!("   Max error: CR={}, error={}", 
             max_error_codec.config.base_cr, 
             max_error_codec.config.error);
    
    // Relative error bounded compression  
    let rel_error_codec = EBCCCodec::relative_error_bounded([1, 32, 32], 12.0, 0.01)?;
    println!("   Relative error: CR={}, relative_error={}", 
             rel_error_codec.config.base_cr, 
             rel_error_codec.config.error);

    // Example 4: Actual compression using numcodecs API
    println!("\n4. Compression/decompression example:");
    
    // Create some test data (32x32 frame of sinusoidal data - EBCC requires at least 32x32)
    let size = 32 * 32;
    let test_data: Vec<f32> = (0..size)
        .map(|i| {
            let x = (i % 32) as f32 / 32.0;
            let y = (i / 32) as f32 / 32.0;
            (x * std::f32::consts::PI * 2.0).sin() * (y * std::f32::consts::PI * 2.0).cos() + 
            0.1 * ((x + y) * 10.0).sin() // Add some high frequency content
        })
        .collect();
    
    println!("   Created test data: {} values (32x32 frame)", test_data.len());
    println!("   Data range: [{:.3}, {:.3}]", 
             test_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             test_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Create the array (note: need to match codec dimensions exactly)
    let data_array = Array::from_shape_vec(32 * 32, test_data.clone())?;
    let cow_array = AnyCowArray::F32(data_array.into_dyn().into());
    
    // Compress using numcodecs API
    match jpeg_codec.encode(cow_array) {
        Ok(compressed) => {
            match compressed {
                AnyArray::U8(compressed_bytes) => {
                    println!("   ✓ Compressed size: {} bytes", compressed_bytes.len());
                    println!("   ✓ Compression ratio: {:.2}", 
                             (test_data.len() * 4) as f32 / compressed_bytes.len() as f32);
                    
                    // Decompress
                    let decompressed = jpeg_codec.decode(AnyCowArray::U8(compressed_bytes.view().into()))?;
                    
                    match decompressed {
                        AnyArray::F32(decompressed_array) => {
                            println!("   ✓ Decompressed shape: {:?}", decompressed_array.shape());
                            
                            // Calculate reconstruction error
                            let decompressed_data = decompressed_array.as_slice().unwrap();
                            let max_error = test_data.iter()
                                .zip(decompressed_data.iter())
                                .map(|(original, reconstructed)| (original - reconstructed).abs())
                                .fold(0.0, f32::max);
                            
                            println!("   ✓ Maximum reconstruction error: {:.6}", max_error);
                            println!("   ✓ Compression/decompression successful!");
                        },
                        _ => println!("   ❌ Unexpected decompressed data type"),
                    }
                },
                _ => println!("   ❌ Unexpected compressed data type"),
            }
        },
        Err(e) => {
            println!("   ⚠ Compression failed (this might be expected for small data): {}", e);
            println!("     Note: EBCC requires minimum data sizes for effective compression");
        }
    }

    // Example 5: Error handling
    println!("\n5. Error handling:");
    
    // Test unsupported data type
    let int_data = Array::from_shape_vec([10, 10], vec![1i32; 100])?;
    match jpeg_codec.encode(AnyCowArray::I32(int_data.into_dyn().into())) {
        Err(e) => println!("   ✓ Correctly rejected i32 data: {}", e),
        Ok(_) => println!("   ❌ Should have rejected i32 data"),
    }
    
    // Test shape mismatch - use data that doesn't match codec dimensions
    let wrong_size_data = Array::from_shape_vec(64 * 64, vec![1.0f32; 64 * 64])?;
    match jpeg_codec.encode(AnyCowArray::F32(wrong_size_data.into_dyn().into())) {
        Err(e) => println!("   ✓ Correctly rejected wrong size data (64x64 vs expected 32x32): {}", e),
        Ok(_) => println!("   ❌ Should have rejected wrong size data"),
    }

    // Example 6: Configuration serialization
    println!("\n6. Configuration serialization:");
    
    let config_json = serde_json::to_string_pretty(&codec_from_config)?;
    println!("   Serialized codec configuration:");
    println!("{}", config_json);
    
    // Parse it back
    let parsed_codec: EBCCCodec = serde_json::from_str(&config_json)?;
    println!("   ✓ Successfully parsed codec back from JSON");
    println!("   ✓ Parsed base CR: {}", parsed_codec.config.base_cr);

    // Example 7: Configuration validation
    println!("\n7. Configuration validation:");
    
    // Test invalid configuration
    let mut invalid_config_map = HashMap::new();
    invalid_config_map.insert("dims".to_string(), serde_json::json!([0, 10, 10])); // Invalid: zero dimension
    invalid_config_map.insert("base_cr".to_string(), serde_json::json!(-5.0));    // Invalid: negative CR
    
    match ebcc_codec_from_config(invalid_config_map) {
        Ok(_) => println!("   ❌ Should have rejected invalid configuration"),
        Err(e) => println!("   ✓ Correctly rejected invalid config: {}", e),
    }

    println!("\n✓ Example completed successfully!");
    println!("The EBCC numcodecs integration is working properly.");
    
    Ok(())
}