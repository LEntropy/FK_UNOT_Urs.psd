//! Direct 8x8 DCT-II / DCT-III (the same transform JPEG itself uses), via a
//! precomputed basis matrix rather than a general-purpose FFT-based DCT
//! crate. 8x8 is small and fixed-size, so a naive separable matrix multiply
//! (D = C * block * C^T, inverse = C^T * D * C) is simple, fast enough, and
//! -- unlike pulling in a library -- guarantees the exact same basis is used
//! for both embedding and detection, which a coefficient-relation watermark
//! depends on.

pub const BLOCK_SIZE: usize = 8;
pub type Block = [[f32; BLOCK_SIZE]; BLOCK_SIZE];

/// C[u][x] = alpha(u) * cos(pi/8 * (x + 0.5) * u)
fn basis_matrix() -> Block {
    let mut c = [[0f32; BLOCK_SIZE]; BLOCK_SIZE];
    for u in 0..BLOCK_SIZE {
        let alpha = if u == 0 {
            (1.0 / BLOCK_SIZE as f32).sqrt()
        } else {
            (2.0 / BLOCK_SIZE as f32).sqrt()
        };
        for x in 0..BLOCK_SIZE {
            let angle = std::f32::consts::PI / BLOCK_SIZE as f32 * (x as f32 + 0.5) * u as f32;
            c[u][x] = alpha * angle.cos();
        }
    }
    c
}

fn matmul(a: &Block, b: &Block) -> Block {
    let mut out = [[0f32; BLOCK_SIZE]; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        for j in 0..BLOCK_SIZE {
            let mut sum = 0f32;
            for k in 0..BLOCK_SIZE {
                sum += a[i][k] * b[k][j];
            }
            out[i][j] = sum;
        }
    }
    out
}

fn transpose(a: &Block) -> Block {
    let mut out = [[0f32; BLOCK_SIZE]; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        for j in 0..BLOCK_SIZE {
            out[j][i] = a[i][j];
        }
    }
    out
}

pub struct Dct8x8 {
    c: Block,
    ct: Block,
}

impl Dct8x8 {
    pub fn new() -> Self {
        let c = basis_matrix();
        let ct = transpose(&c);
        Self { c, ct }
    }

    /// Forward DCT-II: spatial-domain 8x8 block -> frequency-domain coefficients.
    pub fn forward(&self, block: &Block) -> Block {
        matmul(&matmul(&self.c, block), &self.ct)
    }

    /// Inverse DCT (DCT-III): frequency-domain coefficients -> spatial-domain block.
    pub fn inverse(&self, coeffs: &Block) -> Block {
        matmul(&matmul(&self.ct, coeffs), &self.c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_then_inverse_round_trips() {
        let dct = Dct8x8::new();
        let mut block = [[0f32; BLOCK_SIZE]; BLOCK_SIZE];
        let mut v = 0f32;
        for row in block.iter_mut() {
            for cell in row.iter_mut() {
                *cell = v;
                v += 3.7;
            }
        }

        let coeffs = dct.forward(&block);
        let recovered = dct.inverse(&coeffs);

        for i in 0..BLOCK_SIZE {
            for j in 0..BLOCK_SIZE {
                assert!(
                    (recovered[i][j] - block[i][j]).abs() < 1e-3,
                    "mismatch at ({i},{j}): {} vs {}",
                    recovered[i][j],
                    block[i][j]
                );
            }
        }
    }
}
