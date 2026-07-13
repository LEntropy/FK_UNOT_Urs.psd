use clap::{Parser, Subcommand};
use image::{ImageBuffer, Rgb, RgbImage};
use rust_core::robustness::named_transforms;
use rust_core::variants::{generate_variants, DELIVERY_VARIANTS};
use rust_core::watermark::{apply_new_y, rgb_to_y, Watermarker};

#[derive(Parser)]
#[command(name = "rust-core")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed a hex-encoded payload as an invisible watermark.
    Embed {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        /// Hex string, e.g. "deadbeef" (32 bits). Bit count must divide evenly
        /// into (width/8)*(height/8) for even redundancy, but any size works.
        #[arg(long)]
        payload_hex: String,
        /// Coefficient separation to enforce (higher = more robust, more visible).
        #[arg(long, default_value_t = 24.0)]
        strength: f32,
    },
    /// Recover a watermark payload and report per-bit confidence.
    Detect {
        #[arg(long)]
        input: String,
        /// Number of bits to recover (must match what was embedded).
        #[arg(long)]
        bits: usize,
        /// If given, also prints Hamming distance vs this expected hex payload.
        #[arg(long)]
        expected_hex: Option<String>,
    },
    /// Embed a payload, then apply JPEG/resize transforms (matching
    /// ml-engine's robustness_test.py exactly) and report bit-error-rate
    /// after each one.
    Robustness {
        #[arg(long)]
        input: String,
        #[arg(long)]
        payload_hex: String,
        #[arg(long, default_value_t = 24.0)]
        strength: f32,
    },
    /// Sign and embed a C2PA manifest (self-signed identity -- see
    /// src/c2pa_manifest.rs for why, and its limits).
    C2paSign {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        /// File extension or MIME type, e.g. "jpg", "png", "image/jpeg".
        #[arg(long)]
        format: String,
        #[arg(long, default_value = "DONTAI protected artwork")]
        title: String,
        /// JSON string embedded as a custom "com.dontai.ownership" assertion
        /// -- in the real pipeline this is where blockchain-svc's
        /// contentHash/txHash would go.
        #[arg(long, default_value = "{}")]
        ownership_json: String,
    },
    /// Read back an embedded C2PA manifest and report its contents +
    /// validation status.
    C2paVerify {
        #[arg(long)]
        input: String,
        #[arg(long)]
        format: String,
    },
    /// Generate Delivery Gateway resolution variants from an already-
    /// protected image, tagging each with whether the upstream protection
    /// is empirically known to survive at that scale.
    Variants {
        #[arg(long)]
        input: String,
        /// Directory to write variant files into (created if missing).
        #[arg(long)]
        out_dir: String,
    },
}

fn hex_to_bits(hex: &str) -> Vec<bool> {
    let mut bits = Vec::with_capacity(hex.len() * 4);
    for c in hex.chars() {
        let nibble = c.to_digit(16).expect("invalid hex character");
        for i in (0..4).rev() {
            bits.push((nibble >> i) & 1 == 1);
        }
    }
    bits
}

fn bits_to_hex(bits: &[bool]) -> String {
    let mut hex = String::with_capacity(bits.len() / 4);
    for chunk in bits.chunks(4) {
        let mut nibble = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            if b {
                nibble |= 1 << (3 - i);
            }
        }
        hex.push(std::char::from_digit(nibble, 16).unwrap());
    }
    hex
}

/// Embeds `payload` into `img`'s luminance and returns the watermarked RGB image.
fn embed_into_rgb(img: &RgbImage, payload: &[bool], strength: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let (width, height) = (width as usize, height as usize);

    let mut y_plane: Vec<f32> = img.pixels().map(|p| rgb_to_y(p[0], p[1], p[2])).collect();
    Watermarker::new(strength).embed(&mut y_plane, width, height, payload);

    let mut out: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);
    for (i, px) in img.pixels().enumerate() {
        let (r, g, b) = apply_new_y(px[0], px[1], px[2], y_plane[i]);
        let x = (i % width) as u32;
        let y = (i / width) as u32;
        out.put_pixel(x, y, Rgb([r, g, b]));
    }
    out
}

/// Shared by `detect` and `robustness`: extract Y plane, run detection,
/// return (recovered_bits, avg_confidence, min_confidence).
fn detect_from_rgb(img: &RgbImage, bits: usize) -> (Vec<bool>, f32, f32) {
    let (width, height) = img.dimensions();
    let (width, height) = (width as usize, height as usize);
    let y_plane: Vec<f32> = img.pixels().map(|p| rgb_to_y(p[0], p[1], p[2])).collect();

    let wm = Watermarker::new(24.0); // strength only affects embed; detect just reads coefficient order
    let (recovered_bits, confidence) = wm.detect(&y_plane, width, height, bits);
    let avg_conf: f32 = confidence.iter().sum::<f32>() / confidence.len() as f32;
    let min_conf = confidence.iter().cloned().fold(f32::INFINITY, f32::min);
    (recovered_bits, avg_conf, min_conf)
}

fn bit_error_rate(recovered: &[bool], expected: &[bool]) -> f32 {
    let errors = recovered.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
    100.0 * errors as f32 / recovered.len() as f32
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Embed {
            input,
            output,
            payload_hex,
            strength,
        } => {
            let payload = hex_to_bits(&payload_hex);
            let img = image::open(&input).expect("failed to open input image").to_rgb8();
            let out = embed_into_rgb(&img, &payload, strength);
            out.save(&output).expect("failed to save output image");
            println!("[embed] payload={payload_hex} ({} bits) strength={strength} -> {output}", payload.len());
        }

        Commands::Detect {
            input,
            bits,
            expected_hex,
        } => {
            let img = image::open(&input).expect("failed to open input image").to_rgb8();
            let (recovered_bits, avg_conf, min_conf) = detect_from_rgb(&img, bits);
            let hex = bits_to_hex(&recovered_bits);

            println!("[detect] recovered={hex} avg_confidence={avg_conf:.3} min_confidence={min_conf:.3}");

            if let Some(expected) = expected_hex {
                let expected_bits = hex_to_bits(&expected);
                let ber = bit_error_rate(&recovered_bits, &expected_bits);
                println!("[detect] bit error rate vs expected: {ber:.1}%");
            }
        }

        Commands::Robustness {
            input,
            payload_hex,
            strength,
        } => {
            let payload = hex_to_bits(&payload_hex);
            let img = image::open(&input).expect("failed to open input image").to_rgb8();
            let watermarked = embed_into_rgb(&img, &payload, strength);

            println!("{:<16} {:>12} {:>14} {:>14}", "transform", "ber(%)", "avg_conf", "min_conf");
            for (name, transform) in named_transforms() {
                let transformed = transform(&watermarked);
                let (recovered_bits, avg_conf, min_conf) = detect_from_rgb(&transformed, payload.len());
                let ber = bit_error_rate(&recovered_bits, &payload);
                println!("{name:<16} {ber:>11.1}% {avg_conf:>14.3} {min_conf:>14.3}");
            }
        }

        Commands::C2paSign {
            input,
            output,
            format,
            title,
            ownership_json,
        } => {
            let input_bytes = std::fs::read(&input).expect("failed to read input file");
            let ownership: serde_json::Value =
                serde_json::from_str(&ownership_json).expect("--ownership-json must be valid JSON");

            let signed = rust_core::c2pa_manifest::sign_and_embed(
                &input_bytes,
                &format,
                &title,
                "com.dontai.ownership",
                &ownership,
            )
            .expect("failed to sign/embed C2PA manifest");

            std::fs::write(&output, &signed).expect("failed to write output file");
            println!("[c2pa-sign] embedded manifest, title={title:?} -> {output} ({} bytes)", signed.len());
        }

        Commands::C2paVerify { input, format } => {
            let bytes = std::fs::read(&input).expect("failed to read input file");
            let result = rust_core::c2pa_manifest::verify(&bytes, &format).expect("failed to read C2PA manifest");

            println!("[c2pa-verify] manifest:\n{}", result.manifest_json);
            match result.validation_issues {
                None => println!("[c2pa-verify] validation: OK, no issues reported"),
                Some(issues) => {
                    println!("[c2pa-verify] validation reported {} issue(s):", issues.len());
                    for issue in issues {
                        println!("  - {issue}");
                    }
                }
            }
        }

        Commands::Variants { input, out_dir } => {
            let img = image::open(&input).expect("failed to open input image").to_rgb8();
            std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

            let results = generate_variants(&img, DELIVERY_VARIANTS);
            println!("{:<24} {:>10} {:>10} {:>8} {:<28}", "variant", "width", "height", "scale", "protection status");
            for result in &results {
                println!(
                    "{:<24} {:>10} {:>10} {:>7.2}x {:<28}",
                    result.name, result.width, result.height, result.scale_vs_source, result.protection_status.label()
                );
                let out_path = format!("{out_dir}/{}.png", result.name);
                result.image.save(&out_path).expect("failed to save variant");
            }
            println!("[variants] wrote {} variant(s) to {out_dir}", results.len());
        }
    }
}
