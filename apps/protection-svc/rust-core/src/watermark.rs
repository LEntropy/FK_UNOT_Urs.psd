//! Invisible watermark: embeds a fixed-length bit payload into the
//! luminance (Y) channel's mid-frequency DCT coefficients, redundantly
//! across many 8x8 blocks, recovered by majority vote.
//!
//! Design choices and why (contrast with ml-engine's approach, see
//! `apps/protection-svc/ml-engine/README.md`):
//! - Operates on luminance only, not RGB directly: JPEG itself subsamples
//!   chrominance more aggressively than luminance, so encoding into chroma
//!   would be destroyed by compression the watermark is specifically meant
//!   to survive.
//! - Coefficient-*relation* encoding (is coeff_a > coeff_b?) rather than
//!   absolute magnitude: relative order between two coefficients is far
//!   more stable under quantization/requantization than either coefficient's
//!   absolute value, which is exactly what JPEG's own compression does to
//!   DCT coefficients.
//! - Redundant across many blocks + majority vote at detection: single-block
//!   encoding would be fragile to any one block's damage; spreading each
//!   payload bit across every Nth block and voting is a simple, real
//!   error-correction mechanism.
//!
//! This is a mechanism demo, not a production watermarking scheme -- no
//! synchronization/resampling recovery is implemented, so block-grid
//! misalignment (e.g. from resizing) is a known, tested-for weakness (see
//! `tests/robustness.rs`).

use crate::dct::{Block, Dct8x8, BLOCK_SIZE};

/// Two mid-frequency coefficient positions whose *relative* magnitude
/// encodes one bit. Mid-frequency (not DC, not highest-frequency) is the
/// classic choice: low frequencies carry visible image energy (perturbing
/// them is visible), high frequencies are exactly what JPEG quantizes away
/// first (perturbing them doesn't survive compression).
const POS_A: (usize, usize) = (3, 2);
const POS_B: (usize, usize) = (2, 3);

pub struct Watermarker {
    dct: Dct8x8,
    /// How far apart to force the two coefficients (embedding strength).
    /// Larger = more robust, more visible/lossy.
    strength: f32,
}

impl Watermarker {
    pub fn new(strength: f32) -> Self {
        Self {
            dct: Dct8x8::new(),
            strength,
        }
    }

    /// Embeds `payload_bits` into the Y channel of `y_plane` (row-major,
    /// `width` x `height`, values in 0..=255 as f32). Modifies in place.
    /// Blocks beyond a multiple of 8 in either dimension are left untouched.
    pub fn embed(&self, y_plane: &mut [f32], width: usize, height: usize, payload_bits: &[bool]) {
        let blocks_w = width / BLOCK_SIZE;
        let blocks_h = height / BLOCK_SIZE;
        let mut block_index = 0usize;

        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                let bit = payload_bits[block_index % payload_bits.len()];
                let mut block = read_block(y_plane, width, bx, by);
                let mut coeffs = self.dct.forward(&block);
                set_bit(&mut coeffs, bit, self.strength);
                block = self.dct.inverse(&coeffs);
                write_block(y_plane, width, bx, by, &block);
                block_index += 1;
            }
        }
    }

    /// Recovers a `payload_len`-bit payload from `y_plane` via majority vote
    /// across all blocks assigned to each bit position. Returns the
    /// recovered bits plus, per bit, the vote margin (how many blocks agreed
    /// with the majority) as a rough confidence signal.
    pub fn detect(&self, y_plane: &[f32], width: usize, height: usize, payload_len: usize) -> (Vec<bool>, Vec<f32>) {
        let blocks_w = width / BLOCK_SIZE;
        let blocks_h = height / BLOCK_SIZE;

        let mut ones = vec![0u32; payload_len];
        let mut totals = vec![0u32; payload_len];
        let mut block_index = 0usize;

        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                let block = read_block(y_plane, width, bx, by);
                let coeffs = self.dct.forward(&block);
                let bit = read_bit(&coeffs);

                let slot = block_index % payload_len;
                totals[slot] += 1;
                if bit {
                    ones[slot] += 1;
                }
                block_index += 1;
            }
        }

        let mut bits = Vec::with_capacity(payload_len);
        let mut confidence = Vec::with_capacity(payload_len);
        for i in 0..payload_len {
            let total = totals[i].max(1);
            let frac_one = ones[i] as f32 / total as f32;
            bits.push(frac_one > 0.5);
            confidence.push((frac_one - 0.5).abs() * 2.0); // 0 = coin flip, 1 = unanimous
        }
        (bits, confidence)
    }
}

fn set_bit(coeffs: &mut Block, bit: bool, strength: f32) {
    let a = coeffs[POS_A.0][POS_A.1];
    let b = coeffs[POS_B.0][POS_B.1];
    let diff = a - b;
    let target_diff = if bit { strength } else { -strength };
    let delta = (target_diff - diff) / 2.0;
    coeffs[POS_A.0][POS_A.1] = a + delta;
    coeffs[POS_B.0][POS_B.1] = b - delta;
}

fn read_bit(coeffs: &Block) -> bool {
    coeffs[POS_A.0][POS_A.1] > coeffs[POS_B.0][POS_B.1]
}

fn read_block(plane: &[f32], width: usize, bx: usize, by: usize) -> Block {
    let mut block = [[0f32; BLOCK_SIZE]; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        for j in 0..BLOCK_SIZE {
            let x = bx * BLOCK_SIZE + j;
            let y = by * BLOCK_SIZE + i;
            block[i][j] = plane[y * width + x];
        }
    }
    block
}

fn write_block(plane: &mut [f32], width: usize, bx: usize, by: usize, block: &Block) {
    for i in 0..BLOCK_SIZE {
        for j in 0..BLOCK_SIZE {
            let x = bx * BLOCK_SIZE + j;
            let y = by * BLOCK_SIZE + i;
            plane[y * width + x] = block[i][j];
        }
    }
}

/// ITU-R BT.601 RGB <-> YCbCr, matching what JPEG itself uses internally --
/// deliberate, so the watermark's notion of "luminance" lines up with the
/// channel JPEG actually protects most.
pub fn rgb_to_y(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Replaces the Y channel of an RGB pixel while preserving Cb/Cr (color),
/// so only luminance carries the watermark. Reconstructs from (new_y, cb,
/// cr) directly rather than shifting each RGB channel, so chrominance is
/// exactly preserved regardless of how much Y changed.
pub fn apply_new_y(r: u8, g: u8, b: u8, new_y: f32) -> (u8, u8, u8) {
    let cb = -0.168736 * r as f32 - 0.331264 * g as f32 + 0.5 * b as f32;
    let cr = 0.5 * r as f32 - 0.418688 * g as f32 - 0.081312 * b as f32;

    let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    let new_r = new_y + 1.402 * cr;
    let new_g = new_y - 0.344136 * cb - 0.714136 * cr;
    let new_b = new_y + 1.772 * cb;
    (clamp(new_r), clamp(new_g), clamp(new_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_then_detect_recovers_payload_exactly() {
        let width = 64;
        let height = 64;
        let mut y_plane = vec![128f32; width * height];
        // Give blocks some texture instead of flat gray -- flat blocks have
        // all-zero AC coefficients, an unrealistic edge case.
        for (i, v) in y_plane.iter_mut().enumerate() {
            *v = 100.0 + (i % 50) as f32;
        }

        let payload = vec![true, false, true, true, false, false, true, false];
        let wm = Watermarker::new(24.0);
        wm.embed(&mut y_plane, width, height, &payload);

        let (recovered, confidence) = wm.detect(&y_plane, width, height, payload.len());
        assert_eq!(recovered, payload);
        for c in confidence {
            assert!(c > 0.9, "expected high confidence on a clean round-trip, got {c}");
        }
    }
}
