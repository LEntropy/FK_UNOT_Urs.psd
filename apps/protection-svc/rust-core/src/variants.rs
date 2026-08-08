//! Resolution-variant generation for Delivery Gateway (`PROJECT_DESIGN.md`
//! section 3-5: anonymous users get a 1280px variant, logged-in users get
//! 2048px, grid views get a small thumbnail).
//!
//! The important part isn't the resizing itself (that's a few lines with
//! the `image` crate) -- it's tagging each variant with whether the
//! protection mechanisms upstream (ml-engine's cloak, this crate's
//! watermark) can actually be trusted to survive at that variant's scale.
//! Both mechanisms were independently measured (see `ml-engine/README.md`
//! and this crate's README) to hold up well at 0.5x of the protected
//! image's resolution and to fail hard at 0.25x, for two unrelated reasons
//! (an information floor for the ML cloak, block-grid misalignment for the
//! watermark). Nothing was measured *between* those two points, so this
//! reports three states, not a false-precision gradient:
//!
//! - Safe (scale >= 0.5): both mechanisms empirically hold up here.
//! - Unknown (0.25 <= scale < 0.5): never tested at this exact range --
//!   don't claim either way.
//! - Unsafe (scale <= 0.25): both mechanisms empirically fail here.

use image::imageops::FilterType;
use image::RgbImage;

pub struct VariantSpec {
    pub name: &'static str,
    pub max_dimension: u32,
}

/// Matches PROJECT_DESIGN.md section 3-5's named variants.
pub const DELIVERY_VARIANTS: &[VariantSpec] = &[
    VariantSpec { name: "public_preview_2048", max_dimension: 2048 },
    VariantSpec { name: "public_preview_1280", max_dimension: 1280 },
    VariantSpec { name: "grid_thumbnail_512", max_dimension: 512 },
    VariantSpec { name: "grid_thumbnail_150", max_dimension: 150 },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionStatus {
    /// scale >= 0.5 -- both mechanisms empirically hold up here.
    Safe,
    /// 0.25 <= scale < 0.5 -- not measured; don't claim either way.
    Unknown,
    /// scale <= 0.25 -- both mechanisms empirically fail here.
    Unsafe,
}

impl ProtectionStatus {
    fn from_scale(scale: f32) -> Self {
        if scale >= 0.5 {
            ProtectionStatus::Safe
        } else if scale > 0.25 {
            ProtectionStatus::Unknown
        } else {
            ProtectionStatus::Unsafe
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ProtectionStatus::Safe => "SAFE",
            ProtectionStatus::Unknown => "UNKNOWN (untested range)",
            ProtectionStatus::Unsafe => "UNSAFE (protection likely void)",
        }
    }
}

pub struct VariantResult {
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
    pub scale_vs_source: f32,
    pub protection_status: ProtectionStatus,
    pub image: RgbImage,
}

/// Resizes `source` (the already-protected image -- cloak + watermark
/// applied upstream) down to each spec in `specs`, preserving aspect ratio.
/// Specs whose max_dimension is >= the source's own size are skipped (no
/// upscaling -- the source itself already serves that tier).
pub fn generate_variants(source: &RgbImage, specs: &[VariantSpec]) -> Vec<VariantResult> {
    let (src_w, src_h) = source.dimensions();
    let src_max_dim = src_w.max(src_h);

    specs
        .iter()
        .filter(|spec| spec.max_dimension < src_max_dim)
        .map(|spec| {
            let scale = spec.max_dimension as f32 / src_max_dim as f32;
            let (w, h) = if src_w >= src_h {
                (spec.max_dimension, (src_h as f32 * scale).round() as u32)
            } else {
                ((src_w as f32 * scale).round() as u32, spec.max_dimension)
            };

            let resized = image::imageops::resize(source, w.max(1), h.max(1), FilterType::Lanczos3);

            VariantResult {
                name: spec.name,
                width: w,
                height: h,
                scale_vs_source: scale,
                protection_status: ProtectionStatus::from_scale(scale),
                image: resized,
            }
        })
        .collect()
}

/// Feed/gallery thumbnail resolution, generated from the *original*
/// (pre-protection) image rather than the protected one. Unlike
/// DELIVERY_VARIANTS above, the defense here isn't the cloak surviving a
/// resize (this image was never cloaked) -- it's sheer information loss: a
/// LoRA trainer fed a file this small has to upscale it back up to its own
/// training resolution (512px+ for SD1.5, see protection_score.py) first,
/// discarding almost all real detail before training even starts. Picked
/// well below any DELIVERY_VARIANTS entry (smallest there is 150px) for
/// margin.
///
/// NOT empirically validated the way DELIVERY_VARIANTS' Safe/Unknown/Unsafe
/// grid is (that required a real measured LoRA-training run per scale,
/// see this module's own doc above) -- this is a reasoned default, not a
/// proven bound. Treat "raises the bar" as the honest claim here, not
/// "prevents training" -- a real score_protection-style run at this exact
/// resolution would be needed before claiming more.
pub const FEED_THUMBNAIL_MAX_DIMENSION: u32 = 128;

/// Downscales `source` to FEED_THUMBNAIL_MAX_DIMENSION on its long edge,
/// preserving aspect ratio. Unlike generate_variants, never skips (a
/// feed thumbnail is always wanted) -- returns the source unchanged if it's
/// already smaller than the target (no upscaling, same reasoning as
/// generate_variants).
pub fn generate_feed_thumbnail(source: &RgbImage) -> RgbImage {
    let (src_w, src_h) = source.dimensions();
    let src_max_dim = src_w.max(src_h);
    if src_max_dim <= FEED_THUMBNAIL_MAX_DIMENSION {
        return source.clone();
    }

    let scale = FEED_THUMBNAIL_MAX_DIMENSION as f32 / src_max_dim as f32;
    let (w, h) = if src_w >= src_h {
        (FEED_THUMBNAIL_MAX_DIMENSION, (src_h as f32 * scale).round() as u32)
    } else {
        ((src_w as f32 * scale).round() as u32, FEED_THUMBNAIL_MAX_DIMENSION)
    };

    image::imageops::resize(source, w.max(1), h.max(1), FilterType::Lanczos3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(w: u32, h: u32) -> RgbImage {
        image::ImageBuffer::from_fn(w, h, |x, y| image::Rgb([(x % 256) as u8, (y % 256) as u8, 100]))
    }

    #[test]
    fn skips_variants_that_would_upscale() {
        let source = test_image(960, 720);
        let results = generate_variants(&source, DELIVERY_VARIANTS);

        // public_preview_2048 and public_preview_1280 both exceed the 960px
        // source's largest dimension -- neither should appear.
        assert!(!results.iter().any(|r| r.name == "public_preview_2048"));
        assert!(!results.iter().any(|r| r.name == "public_preview_1280"));
        assert!(results.iter().any(|r| r.name == "grid_thumbnail_512"));
        assert!(results.iter().any(|r| r.name == "grid_thumbnail_150"));
    }

    #[test]
    fn preserves_aspect_ratio() {
        let source = test_image(1000, 500); // 2:1
        let results = generate_variants(&source, &[VariantSpec { name: "half", max_dimension: 500 }]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].width, 500);
        assert_eq!(results[0].height, 250);
    }

    #[test]
    fn tags_protection_status_by_measured_thresholds() {
        // Matches the empirical breakpoints from ml-engine/README.md and
        // this crate's own README.md -- see variants.rs's module doc for
        // why these three bands, not a smooth gradient.
        assert_eq!(ProtectionStatus::from_scale(1.0), ProtectionStatus::Safe);
        assert_eq!(ProtectionStatus::from_scale(0.5), ProtectionStatus::Safe);
        assert_eq!(ProtectionStatus::from_scale(0.4), ProtectionStatus::Unknown);
        assert_eq!(ProtectionStatus::from_scale(0.26), ProtectionStatus::Unknown);
        assert_eq!(ProtectionStatus::from_scale(0.25), ProtectionStatus::Unsafe);
        assert_eq!(ProtectionStatus::from_scale(0.1), ProtectionStatus::Unsafe);
    }

    #[test]
    fn feed_thumbnail_downscales_to_max_dimension_preserving_aspect() {
        let source = test_image(1000, 500); // 2:1
        let thumb = generate_feed_thumbnail(&source);
        assert_eq!(thumb.width(), FEED_THUMBNAIL_MAX_DIMENSION);
        assert_eq!(thumb.height(), FEED_THUMBNAIL_MAX_DIMENSION / 2);
    }

    #[test]
    fn feed_thumbnail_never_upscales() {
        let source = test_image(64, 64);
        let thumb = generate_feed_thumbnail(&source);
        assert_eq!((thumb.width(), thumb.height()), (64, 64));
    }
}
