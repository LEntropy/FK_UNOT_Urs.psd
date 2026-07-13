//! Locks in the robustness characteristics measured manually via
//! `cargo run -- robustness` against a real painting (see rust-core's
//! README): strong against JPEG at any tested quality and moderate resizes,
//! breaks down at aggressive (0.25x) resizing due to block-grid
//! misalignment. Uses a synthetic image so this test is self-contained and
//! doesn't depend on the downloaded painting files in ml-engine/out/.

use image::{ImageBuffer, Rgb, RgbImage};
use rust_core::robustness::{jpeg_recompress, resize_round_trip};
use rust_core::watermark::{apply_new_y, rgb_to_y, Watermarker};

const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const PAYLOAD_HEX: &str = "deadbeefcafef00d"; // 64 bits
const STRENGTH: f32 = 24.0;

fn hex_to_bits(hex: &str) -> Vec<bool> {
    let mut bits = Vec::with_capacity(hex.len() * 4);
    for c in hex.chars() {
        let nibble = c.to_digit(16).unwrap();
        for i in (0..4).rev() {
            bits.push((nibble >> i) & 1 == 1);
        }
    }
    bits
}

fn synthetic_test_image() -> RgbImage {
    // Textured gradient + checkerboard, not flat color -- flat blocks have
    // all-zero AC coefficients, an unrealistic edge case for a DCT watermark.
    ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
        let checker = if (x / 8 + y / 8) % 2 == 0 { 40 } else { 0 };
        let r = ((x * 255 / WIDTH as u32) as u8).saturating_add(checker);
        let g = ((y * 255 / HEIGHT as u32) as u8).saturating_add(checker);
        let b = 128u8.saturating_add(checker);
        Rgb([r, g, b])
    })
}

fn embed(img: &RgbImage, payload: &[bool]) -> RgbImage {
    let mut y_plane: Vec<f32> = img.pixels().map(|p| rgb_to_y(p[0], p[1], p[2])).collect();
    Watermarker::new(STRENGTH).embed(&mut y_plane, WIDTH, HEIGHT, payload);

    let mut out: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
    for (i, px) in img.pixels().enumerate() {
        let (r, g, b) = apply_new_y(px[0], px[1], px[2], y_plane[i]);
        out.put_pixel((i % WIDTH) as u32, (i / WIDTH) as u32, Rgb([r, g, b]));
    }
    out
}

fn bit_error_count(img: &RgbImage, expected: &[bool]) -> usize {
    let y_plane: Vec<f32> = img.pixels().map(|p| rgb_to_y(p[0], p[1], p[2])).collect();
    let (recovered, _confidence) = Watermarker::new(STRENGTH).detect(&y_plane, WIDTH, HEIGHT, expected.len());
    recovered.iter().zip(expected.iter()).filter(|(a, b)| a != b).count()
}

#[test]
fn clean_round_trip_has_zero_bit_errors() {
    let payload = hex_to_bits(PAYLOAD_HEX);
    let img = synthetic_test_image();
    let watermarked = embed(&img, &payload);
    assert_eq!(bit_error_count(&watermarked, &payload), 0);
}

#[test]
fn survives_jpeg_recompression_at_all_tested_qualities() {
    // Exact bit-error counts depend on image content (measured 0/64 on a
    // real painting -- see README -- vs a few bits here on a synthetic
    // checkerboard, whose sharp 8px-period frequency content happens to sit
    // closer to the coefficients this scheme uses). The property this test
    // protects is "JPEG doesn't wreck most of the payload," not "always
    // exactly zero errors on every possible image."
    let payload = hex_to_bits(PAYLOAD_HEX);
    let img = synthetic_test_image();
    let watermarked = embed(&img, &payload);

    for (quality, max_allowed_errors) in [(95, 1), (75, 2), (50, 6)] {
        let recompressed = jpeg_recompress(&watermarked, quality);
        let errors = bit_error_count(&recompressed, &payload);
        assert!(
            errors <= max_allowed_errors,
            "expected <= {max_allowed_errors} bit errors at JPEG q{quality}, got {errors}"
        );
    }
}

#[test]
fn survives_moderate_resize_but_not_aggressive_resize() {
    let payload = hex_to_bits(PAYLOAD_HEX);
    let img = synthetic_test_image();
    let watermarked = embed(&img, &payload);

    let moderate = resize_round_trip(&watermarked, 0.5);
    let moderate_errors = bit_error_count(&moderate, &payload);
    assert!(moderate_errors <= 2, "expected near-zero errors at 0.5x resize, got {moderate_errors}");

    // Known limitation (see README): aggressive downscaling misaligns the
    // 8x8 block grid the watermark depends on. This assertion documents
    // that the weakness exists, not a specific bit-error target -- it
    // should fail loudly if a future change accidentally "fixes" this
    // without anyone noticing (which would be surprising and worth
    // investigating, not silently accepting).
    let aggressive = resize_round_trip(&watermarked, 0.25);
    let aggressive_errors = bit_error_count(&aggressive, &payload);
    assert!(
        aggressive_errors > 5,
        "expected 0.25x resize to break the watermark (known limitation); got only {aggressive_errors} errors -- if this now passes reliably, update the README, don't just loosen this assertion"
    );
}
