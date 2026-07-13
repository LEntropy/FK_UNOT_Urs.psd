//! Mirrors `apps/protection-svc/ml-engine/src/robustness_test.py`'s
//! transform set exactly, so the watermark's numbers and the ML cloak's
//! numbers are directly comparable, not measured differently and compared
//! apples-to-oranges.

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::RgbImage;

pub fn jpeg_recompress(img: &RgbImage, quality: u8) -> RgbImage {
    let mut buf = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder.encode_image(img).expect("jpeg encode failed");
    image::load_from_memory(&buf).expect("jpeg decode failed").to_rgb8()
}

pub fn resize_round_trip(img: &RgbImage, scale: f32) -> RgbImage {
    let (w, h) = img.dimensions();
    let small_w = ((w as f32) * scale).max(1.0) as u32;
    let small_h = ((h as f32) * scale).max(1.0) as u32;
    let small = image::imageops::resize(img, small_w, small_h, FilterType::CatmullRom);
    image::imageops::resize(&small, w, h, FilterType::CatmullRom)
}

pub fn sns_pipeline(img: &RgbImage) -> RgbImage {
    jpeg_recompress(&resize_round_trip(img, 0.5), 75)
}

pub fn named_transforms() -> Vec<(&'static str, fn(&RgbImage) -> RgbImage)> {
    vec![
        ("none", |img| img.clone()),
        ("jpeg_q95", |img| jpeg_recompress(img, 95)),
        ("jpeg_q75", |img| jpeg_recompress(img, 75)),
        ("jpeg_q50", |img| jpeg_recompress(img, 50)),
        ("resize_0.5x", |img| resize_round_trip(img, 0.5)),
        ("resize_0.25x", |img| resize_round_trip(img, 0.25)),
        ("sns_pipeline", sns_pipeline),
    ]
}
