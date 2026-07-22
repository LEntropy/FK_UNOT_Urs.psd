//! Locks in the current, honestly-documented state of C2PA embedding (see
//! rust-core/README.md's C2PA section for the full investigation): manifest
//! embedding, custom assertions, content-hash data integrity, and the claim
//! signature itself all validate correctly on read-back. The one remaining
//! validation status is `signingCredential.untrusted`, which is the
//! expected, honest result of using a self-signed identity that isn't in
//! any trust list (see `LocalSigner`/`verify()` docs) -- not a bug.

use rust_core::c2pa_manifest::{sign_and_embed, verify, LocalSigner};

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
    let key_path = std::env::temp_dir().join("dontai_c2pa_test_key_readable_manifest.der");

    let signed = sign_and_embed(&input, "png", "test artwork", "com.dontai.ownership", &ownership, &key_path)
        .expect("sign_and_embed should succeed and produce embeddable bytes");
    assert!(signed.len() > input.len(), "signed output should be larger (manifest embedded)");

    let result = verify(&signed, "png").expect("verify should be able to read back the embedded manifest");

    assert!(result.manifest_json.contains("test artwork"));
    assert!(result.manifest_json.contains("com.dontai.ownership"));
    assert!(result.manifest_json.contains("0xdeadbeef"));
    assert!(result.manifest_json.contains(r#""code": "assertion.dataHash.match""#));
}

#[test]
fn claim_signature_validates_correctly_self_signed_cert_flagged_as_untrusted() {
    // See README.md "C2PA manifest" section for the full writeup. This pins
    // the current good state (real cryptographic signature validation
    // passes) so a silent regression doesn't go unremarked.
    let input = synthetic_png();
    let key_path = std::env::temp_dir().join("dontai_c2pa_test_key_claim_signature.der");
    let signed = sign_and_embed(&input, "png", "t", "com.dontai.ownership", &serde_json::json!({}), &key_path)
        .expect("sign_and_embed should succeed");

    let result = verify(&signed, "png").expect("verify should read back the manifest");
    let issues = result.validation_issues.expect("expected validation issues to be present");

    assert!(
        !issues.iter().any(|i| i.contains("claimSignature.mismatch")),
        "claim signature should cryptographically validate; got: {issues:?}"
    );
    assert!(
        issues.iter().any(|i| i.contains("signingCredential.untrusted")),
        "expected only the expected self-signed-cert-untrusted status; got: {issues:?}"
    );
}

#[test]
fn load_or_generate_reuses_the_same_identity_across_calls() {
    // The real bug this fixes: generate() made a brand new keypair every
    // single call, so every image signed by this pipeline would carry a
    // different, unrelated "identity" -- no way to say two DONTAI images
    // share a signer. load_or_generate() must persist and reload instead.
    let key_path = std::env::temp_dir().join(format!(
        "dontai_c2pa_test_key_reuse_{}.der",
        std::process::id() // avoid colliding with a concurrent test run
    ));
    let _ = std::fs::remove_file(&key_path); // start clean regardless of a leftover from a prior run

    let first = LocalSigner::load_or_generate(&key_path);
    let second = LocalSigner::load_or_generate(&key_path);

    assert_eq!(
        first.public_key_hex(),
        second.public_key_hex(),
        "two load_or_generate calls against the same path should be the same identity"
    );

    let _ = std::fs::remove_file(&key_path); // don't leave test key material behind
}

#[test]
fn generate_produces_a_different_identity_each_call() {
    // The behavior load_or_generate exists to fix, pinned so it isn't
    // accidentally "fixed" back the other way -- generate() is still used
    // internally as load_or_generate's fallback when a key file can't be
    // read/written, and that fallback needs to actually generate fresh,
    // not silently reuse process-global state.
    let first = LocalSigner::generate();
    let second = LocalSigner::generate();

    assert_ne!(first.public_key_hex(), second.public_key_hex());
}
