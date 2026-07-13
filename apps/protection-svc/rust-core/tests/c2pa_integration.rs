//! Locks in the current, honestly-documented state of C2PA embedding (see
//! rust-core/README.md's C2PA section for the full investigation): manifest
//! embedding, custom assertions, and content-hash data integrity all work
//! correctly; the claim signature itself does not cryptographically
//! validate on read-back with this crate version's `rust_native_crypto`
//! backend + a custom `Signer` implementation. If a future crate upgrade
//! fixes this, this test should start failing at the last assertion --
//! that's the signal to update README.md and this test together, not to
//! quietly loosen the assertion.

use rust_core::c2pa_manifest::{sign_and_embed, verify};

fn synthetic_png() -> Vec<u8> {
    let img = image::ImageBuffer::from_fn(64, 64, |x, y| {
        image::Rgb([(x * 4) as u8, (y * 4) as u8, 128u8])
    });
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

#[test]
fn sign_embeds_a_readable_manifest_with_correct_assertions() {
    let input = synthetic_png();
    let ownership = serde_json::json!({ "contentHash": "0xdeadbeef", "chain": "polygon-amoy" });

    let signed = sign_and_embed(&input, "png", "test artwork", "com.dontai.ownership", &ownership)
        .expect("sign_and_embed should succeed and produce embeddable bytes");
    assert!(signed.len() > input.len(), "signed output should be larger (manifest embedded)");

    let result = verify(&signed, "png").expect("verify should be able to read back the embedded manifest");

    assert!(result.manifest_json.contains("test artwork"));
    assert!(result.manifest_json.contains("com.dontai.ownership"));
    assert!(result.manifest_json.contains("0xdeadbeef"));
    assert!(result.manifest_json.contains(r#""code": "assertion.dataHash.match""#));
}

#[test]
fn claim_signature_does_not_validate_known_limitation() {
    // See README.md "C2PA manifest" section for the full writeup. This pins
    // the current known-bad state so a silent regression (or a silent,
    // unnoticed *fix*) doesn't go unremarked.
    let input = synthetic_png();
    let signed = sign_and_embed(&input, "png", "t", "com.dontai.ownership", &serde_json::json!({}))
        .expect("sign_and_embed should succeed");

    let result = verify(&signed, "png").expect("verify should read back the manifest");
    let issues = result.validation_issues.expect("expected validation issues to be present");

    assert!(
        issues.iter().any(|i| i.contains("claimSignature.mismatch")),
        "expected the known claimSignature.mismatch issue; got: {issues:?}"
    );
}
