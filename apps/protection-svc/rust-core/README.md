# protection-svc / rust-core — invisible watermark + C2PA PoC

The traceability layer of `PROJECT_DESIGN.md` section 3-3 ("invisible
watermark + robust fingerprint + C2PA manifest"). This PoC implements the
watermark and a working (if imperfect -- see below) C2PA manifest;
resolution-variant generation is not built yet (see
`apps/protection-svc/INTEGRATION.md` for the full pipeline this fits into
and what's still missing).

## How it works

Embeds a fixed-length bit payload into the luminance (Y) channel's
mid-frequency DCT coefficients, redundantly across every 8x8 block (matching
JPEG's own block grid on purpose), recovered by majority vote across all
blocks assigned to each bit. Full rationale in `src/watermark.rs`'s doc
comment. Short version:

- **Luminance only**: JPEG subsamples chrominance harder than luminance, so
  encoding into color channels would be destroyed by the exact compression
  the watermark needs to survive.
- **Coefficient *relation*, not magnitude**: whether coefficient A is bigger
  than coefficient B survives quantization far better than either
  coefficient's absolute value does.
- **8x8 blocks, same grid as JPEG**: not a coincidence -- it's why this
  survives JPEG recompression as well as it does (see results below).
- **Redundancy + majority vote**: each payload bit is repeated across every
  Nth block; detection votes across all of them, a simple, real form of
  error correction.

This is a mechanism demo, not a production watermarking scheme -- there is
no synchronization/resampling recovery, so block-grid misalignment (e.g.
from an aggressive resize) is a real, tested-for weakness, not an oversight.

## Quick start

```bash
cargo build
cargo test

cargo run -- embed --input ../ml-engine/out/real/starry_night.jpg \
  --output watermarked.png --payload-hex deadbeefcafef00d --strength 24.0

cargo run -- detect --input watermarked.png --bits 64 --expected-hex deadbeefcafef00d
# [detect] recovered=deadbeefcafef00d avg_confidence=1.000 min_confidence=1.000
# [detect] bit error rate vs expected: 0.0%

cargo run -- robustness --input ../ml-engine/out/real/starry_night.jpg \
  --payload-hex deadbeefcafef00d --strength 24.0
```

## Measured robustness (real painting, Van Gogh's *Starry Night*, 64-bit payload)

| Transform | bit error rate | avg confidence |
|---|---|---|
| none (baseline) | 0.0% | 1.000 |
| JPEG q95 | 0.0% | 1.000 |
| JPEG q75 | 0.0% | 1.000 |
| JPEG q50 | 0.0% | 0.996 |
| resize 0.5x round-trip | 0.0% | 0.815 |
| resize 0.25x round-trip | **37.5%** | 0.062 |
| sns_pipeline (resize 0.5x + JPEG q75) | 0.0% | 0.648 |

## This is the "different robustness properties" the design doc predicted

`apps/protection-svc/INTEGRATION.md` noted the watermark is "a different
mechanism with different robustness properties than the ML cloak" -- here's
the side-by-side, same paintings, same transforms, from
`apps/protection-svc/ml-engine/README.md`'s numbers:

| Transform | ML cloak (style drift retained) | Watermark (bit error rate) |
|---|---|---|
| JPEG q95 | 95% retained | 0.0% errors |
| JPEG q75 | 71% retained | 0.0% errors |
| JPEG q50 | 53% retained | 0.0% errors |
| resize 0.5x | -26% retained (fails, EOT-fixed to +11%) | 0.0% errors |
| resize 0.25x | -148% retained (fails, EOT doesn't fix it) | **37.5% errors (fails)** |
| sns_pipeline | -38% retained (EOT-fixed to -6%) | 0.0% errors |

**The watermark is strictly more JPEG-robust and more resize-robust up to
0.5x than the ML cloak** (which needed EOT training just to survive 0.5x at
all, per `ml-engine/README.md`). **Both mechanisms independently break down
at 0.25x resize** -- but for different, unrelated reasons:

- ML cloak: an *information floor*. At 256x256 -> 64x64, there may not be
  enough pixels left to carry the adversarial signal, no matter how it was
  trained (three separate EOT strategies all failed to fix this -- see
  `ml-engine/README.md`).
- Watermark: a *geometric* failure. Resizing to 0.25x and back doesn't
  preserve the original 8x8 block grid this scheme depends on -- the pixels
  a "block" covers after the round-trip don't correspond to a single
  coherent block that was contiguously encoded, so detection reads mostly
  noise (confidence 0.062, barely above a coin flip).

**Practical implication for asset-service/Delivery Gateway** (also noted in
`apps/protection-svc/INTEGRATION.md`): the two mechanisms have overlapping
but not identical failure zones. Neither one should be assumed to protect
an image at every generated resolution -- a small thumbnail (well past
0.25x of a typical upload) has *neither* the ML cloak's style-confusion nor
this watermark's traceability intact. Fixing this (per-thumbnail-size
watermarking, or a resampling-invariant synchronization scheme) is real
future work, not attempted here.

## Re-validated at 1024x1024: the two failure modes diverge exactly as their explanations predict

`ml-engine/README.md` re-ran its cloak at 1024x1024 and found its 0.25x
resize failure *mostly disappears* at higher base resolution (an
information-floor problem: 1024 * 0.25 = 256px, which has "enough room"
again). Ran the equivalent test here:

```bash
cargo run -- embed --input ../ml-engine/out/real/cloaked_1024.png \
  --output watermarked_1024.png --payload-hex deadbeefcafef00d --strength 24.0
cargo run -- robustness --input watermarked_1024.png --payload-hex deadbeefcafef00d --strength 24.0
```

| Transform | 256px BER | 1024px BER |
|---|---|---|
| resize 0.5x | 0.0% | 0.0% |
| resize 0.25x | 37.5% (fails) | **29.7%** (still fails) |

**The watermark's 0.25x failure barely moves with resolution** -- confirming
it really is a different mechanism than the ML cloak's, exactly as
theorized above. The ML cloak's problem is about how much *information*
survives a resize (more base resolution = more room = better survival). The
watermark's problem is *geometric*: resizing to 0.25x and back doesn't
preserve which pixels belonged to which original 8x8 block, and that
misalignment happens at the same relative severity regardless of how big
the image was to begin with -- 1024->256 misaligns blocks just as thoroughly
as 256->64 does, because the resampling math doesn't care about absolute
scale, only the ratio.

**This is a clean, useful confirmation, not just consistent data:** it means
these two mechanisms need genuinely different fixes, not one shared fix.
Building at a higher base resolution would measurably help the ML cloak's
small-thumbnail problem for free; it would do essentially nothing for the
watermark's, which needs an actual resampling-synchronization scheme (real
future work, still not attempted here) to close.

## C2PA manifest

`src/c2pa_manifest.rs`, using the official `c2pa` crate (v0.89, `rust_native_crypto`
feature). Embeds a manifest with a title, `c2pa.actions` assertion, and a
custom `com.dontai.ownership` assertion -- in the real pipeline this is
where `blockchain-svc`'s `contentHash`/`txHash` would go, tying C2PA
provenance to the on-chain registration (`apps/blockchain-svc/INTEGRATION.md`).

```bash
cargo run -- c2pa-sign --input watermarked.png --output watermarked_c2pa.png \
  --format png --title "Starry Night (DONTAI protected)" \
  --ownership-json '{"contentHash":"0x...","chain":"polygon-amoy"}'

cargo run -- c2pa-verify --input watermarked_c2pa.png --format png
```

### Why a self-signed identity instead of the crate's own signer helpers

`c2pa`'s `create_signer`/`file_io` convenience API requires the `openssl`
feature, which pulls in `openssl` with `vendored` (builds OpenSSL from
source) -- and that build failed here (missing a Perl module the vendored
build script needs, on this Windows/git-bash setup). Rather than fight that
toolchain problem, `LocalSigner` in `src/c2pa_manifest.rs` implements the
crate's `Signer` trait directly against `ed25519-dalek` (already a
transitive dependency of `c2pa` itself) and `rcgen` for the self-signed
certificate -- no vendored C build required. The certificate needs specific
X.509v3 extensions (Extended Key Usage, Key Usage, Basic Constraints,
Authority Key Identifier) or the crate's own cert validation rejects it
during signing with `CoseInvalidCert` -- see the comments in
`LocalSigner::generate()` for exactly which ones and why.

### Known issue: claim signature does not validate on read-back

Signing succeeds, the manifest embeds correctly, and reading it back
recovers the exact title/custom assertion/content-hash JSON with correct
data-integrity checks (`assertion.dataHash.match`, `assertion.hashedURI.match`
all pass). But `Reader`'s `validation_status` also reports:

```
signingCredential.untrusted: signing certificate untrusted   <- expected (self-signed, not from a real CA)
claimSignature.mismatch: claim signature is not valid        <- NOT expected, a real open issue
```

The first is exactly what "self-signed, not CA-issued" should produce and is
not a concern. The second is a genuine problem investigated as follows
before concluding it's outside what's practical to fix in this PoC:

1. **Confirmed the key material is correct in isolation**
   (`examples/debug_keys.rs`): the certificate's embedded public key matches
   the key actually used to sign, and a direct `ed25519-dalek` sign/verify
   round trip (bypassing `c2pa` entirely) succeeds. Rules out a key-mismatch
   or basic-crypto bug.
2. **Not format-specific**: identical failure signing into PNG and JPEG.
3. **Enabling `verify_after_sign`** (a self-check the crate can run
   immediately after signing, off by default) to get a more specific error
   surfaced a *different* failure (`CoseX5ChainMissing`) than what
   `Reader::from_stream` reports on the final embedded file -- even though
   the final file's certificate chain reads back correctly with the right
   `common_name` and serial number. Two different internal checks
   disagreeing about whether the certificate chain is even present points at
   an inconsistency in this signing path with a custom `Signer` under
   `rust_native_crypto` in this crate version, not a bug in our key material
   or signing logic (already ruled out in step 1).

**Conclusion**: this is most likely a rough edge in `c2pa` 0.89.0's
`rust_native_crypto` backend combined with bypassing its own
`create_signer` helpers (which we can't use -- see above). Locked in as a
known-bad state by `tests/c2pa_integration.rs`'s
`claim_signature_does_not_validate_known_limitation` test, so a future
crate upgrade that fixes this will make that test fail loudly (the signal
to update this section, not to quietly delete the test). **Practical
implication**: don't rely on this PoC's C2PA signature as real proof of
authenticity yet -- the manifest and its assertions are correctly
constructed and readable, but the cryptographic guarantee C2PA is supposed
to provide isn't actually verifiable end-to-end here. The blockchain
registration (`blockchain-svc`) remains the mechanism this project can
actually stand behind for provenance today.

## Resolution variants (Delivery Gateway)

`src/variants.rs` generates the named variants `PROJECT_DESIGN.md` section
3-5 calls for (`public_preview_2048`, `public_preview_1280`,
`grid_thumbnail_512`, `grid_thumbnail_150`), resized with Lanczos3,
preserving aspect ratio, never upscaling (a variant whose target size
exceeds the source is skipped rather than blown up).

```bash
cargo run -- embed --input ../ml-engine/out/real/starry_night.jpg \
  --output watermarked.png --payload-hex deadbeefcafef00d --strength 24.0
cargo run -- variants --input watermarked.png --out-dir out_variants
```

```
variant                       width     height    scale protection status
grid_thumbnail_512              512        405    0.53x SAFE
grid_thumbnail_150              150        119    0.16x UNSAFE (protection likely void)
```

(`public_preview_2048`/`1280` don't appear here because the 960px source
used in this example is smaller than both -- the "don't upscale" rule
kicking in correctly, not a bug.)

**The point of this module isn't the resizing -- it's the `protection_status`
tag.** Both upstream protection mechanisms were independently measured to
survive at 0.5x of the protected image's resolution and to fail at 0.25x,
for two unrelated reasons (ml-engine's information floor, this crate's
block-grid misalignment -- see the robustness sections above and in
`ml-engine/README.md`). Nothing was measured *between* those two points, so
`ProtectionStatus` reports three bands, not a smooth invented gradient:

| Band | Scale vs. source | Basis |
|---|---|---|
| `Safe` | >= 0.5 | both mechanisms empirically hold here |
| `Unknown` | 0.25 - 0.5 | never tested at this range -- not claimed either way |
| `Unsafe` | <= 0.25 | both mechanisms empirically fail here |

This turns the "don't assume small thumbnails are protected"
finding from `apps/protection-svc/INTEGRATION.md` into something
asset-service can act on mechanically per generated variant, instead of a
prose warning someone has to remember.

## What this PoC does not do

- No resampling-synchronization recovery for the watermark -- the 0.25x
  resize failure above is the direct consequence.
- C2PA claim signature doesn't validate on read-back -- see the dedicated
  section above.
- Variant generation doesn't re-embed the watermark or C2PA manifest per
  variant -- it resizes the already-protected source as-is. Given the
  `Unsafe`/`Unknown` findings above, a real fix for small thumbnails would
  likely need per-variant re-protection, not just tagging the risk (see
  `apps/protection-svc/INTEGRATION.md`'s options for this, still undecided).
- Payload here is an arbitrary hex string for testing; real usage would
  embed something meaningful (e.g. an asset ID or registry reference) and
  needs a decision on what that identifier actually is and how large it
  needs to be.
