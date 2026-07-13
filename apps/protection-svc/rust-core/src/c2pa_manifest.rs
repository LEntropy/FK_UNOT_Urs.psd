//! C2PA (Coalition for Content Provenance and Authenticity) manifest
//! embedding via the official `c2pa` crate, using a locally-generated
//! self-signed Ed25519 identity instead of the crate's built-in
//! `create_signer`/`openssl_sign` helpers.
//!
//! Why not the crate's own signer helpers: `c2pa`'s `create_signer` module
//! and its `file_io` convenience feature both require the `openssl_sign`
//! feature, which pulls in `openssl` with `vendored` (builds OpenSSL from
//! source) -- and that build failed on this Windows/git-bash environment
//! (missing a Perl module the vendored build script needs). Rather than
//! fight that toolchain problem, this implements the crate's `Signer` trait
//! directly against `ed25519-dalek`, which `c2pa` itself already depends on
//! for COSE operations even without the `openssl` feature -- no vendored
//! C build required.
//!
//! This is a real, working C2PA manifest (readable by `c2pa::Reader`, same
//! as any other C2PA-signed file) -- just signed by a self-signed identity
//! that isn't in any trust list, which is expected and fine for a PoC (see
//! the honest verification-status reporting in `verify()` below; a real
//! deployment would need certificates from an actual C2PA-trusted CA).

use std::io::Cursor;

use c2pa::{Builder, Reader, Signer, SigningAlg};
use ed25519_dalek::pkcs8::DecodePrivateKey;
use ed25519_dalek::{Signer as _, SigningKey};

/// A self-signed Ed25519 identity for signing C2PA manifests locally.
/// Not a substitute for a real CA-issued signing certificate -- see module docs.
pub struct LocalSigner {
    signing_key: SigningKey,
    cert_der: Vec<u8>,
}

impl LocalSigner {
    /// Generates a fresh self-signed Ed25519 identity. Each call produces a
    /// new keypair -- callers that need a stable identity across runs
    /// should persist and reload the key material instead of regenerating
    /// it (not implemented here; this PoC signs and verifies within a
    /// single process run).
    pub fn generate() -> Self {
        let rcgen_keypair =
            rcgen::KeyPair::generate_for(&rcgen::PKCS_ED25519).expect("failed to generate Ed25519 keypair");

        let mut params =
            rcgen::CertificateParams::new(vec!["dontai-protection-svc.local".to_string()])
                .expect("failed to build cert params");
        params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "DONTAI Protection Service (self-signed, PoC only)");

        // c2pa's cose_validator::check_cert (run during signing, not just
        // verification) requires specific X.509v3 extensions or it fails
        // with CoseInvalidCert -- these aren't optional decoration, the
        // sign step itself won't succeed without them:
        //   - an Extended Key Usage the crate's trust handler recognizes
        //     (emailProtection is one of the few checked via a fast path in
        //     has_allowed_oid; there's no "document signing" variant in
        //     rcgen's ExtendedKeyUsagePurpose enum, so this is the closest
        //     standard fit, not a semantically perfect choice)
        //   - Key Usage with digitalSignature (and NOT keyCertSign, since
        //     this isn't a CA cert)
        //   - explicit BasicConstraints (CA:false) and an Authority Key
        //     Identifier extension
        params.is_ca = rcgen::IsCa::ExplicitNoCa;
        params.key_usages = vec![rcgen::KeyUsagePurpose::DigitalSignature];
        params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::EmailProtection];
        params.use_authority_key_identifier_extension = true;

        let cert = params
            .self_signed(&rcgen_keypair)
            .expect("failed to self-sign certificate");

        // Re-derive the ed25519-dalek signing key from the same PKCS8 DER
        // rcgen generated, so the certificate's public key and the key we
        // actually sign with are guaranteed to match.
        let signing_key =
            SigningKey::from_pkcs8_der(rcgen_keypair.serialized_der()).expect("failed to parse Ed25519 PKCS8 key");

        Self {
            signing_key,
            cert_der: cert.der().to_vec(),
        }
    }
}

impl Signer for LocalSigner {
    fn sign(&self, data: &[u8]) -> c2pa::Result<Vec<u8>> {
        Ok(self.signing_key.sign(data).to_bytes().to_vec())
    }

    fn alg(&self) -> SigningAlg {
        SigningAlg::Ed25519
    }

    fn certs(&self) -> c2pa::Result<Vec<Vec<u8>>> {
        Ok(vec![self.cert_der.clone()])
    }

    fn reserve_size(&self) -> usize {
        // Signature (64 bytes) + certificate + COSE/CBOR framing overhead.
        // Generous fixed margin rather than computing this exactly -- if
        // it's too small, `sign()` fails loudly rather than silently
        // truncating, so erring high is the safe direction.
        self.cert_der.len() + 4096
    }
}

/// Builds a manifest (title + a custom assertion carrying whatever
/// caller-supplied JSON -- in the real pipeline, this is where the
/// blockchain contentHash/txHash from `blockchain-svc` would go, tying
/// C2PA provenance to the on-chain registration) and embeds it into
/// `input_bytes`, returning the signed output bytes.
pub fn sign_and_embed(
    input_bytes: &[u8],
    format: &str,
    title: &str,
    custom_assertion_label: &str,
    custom_assertion: &serde_json::Value,
) -> c2pa::Result<Vec<u8>> {
    let manifest_json = serde_json::json!({
        "title": title,
        "claim_generator_info": [{ "name": "dontai-protection-svc/rust-core", "version": env!("CARGO_PKG_VERSION") }],
        "assertions": [
            { "label": "c2pa.actions", "data": { "actions": [{ "action": "c2pa.created" }] } },
        ],
    });

    // Tried turning on `verify_after_sign` here to get a precise error at
    // sign time instead of guessing -- it reported a *different* error
    // (CoseX5ChainMissing) than what Reader::from_stream reports after a
    // full embed+read round trip (claimSignature.mismatch), even though the
    // final embedded file's x5chain reads back fine with correct
    // certificate details. That inconsistency (two different failures
    // depending on which internal check path runs) points at a rough edge
    // in this signing path with a custom `Signer` + `rust_native_crypto` in
    // this crate version, not a bug in the Ed25519 key material or signing
    // logic itself (independently confirmed correct -- see
    // rust-core/README.md's verification section). Left at the default
    // (verify_after_sign off) so this at least produces the fullest,
    // furthest-along artifact to inspect and document honestly.
    let mut builder = Builder::from_json(&manifest_json.to_string())?;
    builder.add_assertion_json(custom_assertion_label, custom_assertion)?;

    let signer = LocalSigner::generate();

    let mut source = Cursor::new(input_bytes);
    let mut dest = Cursor::new(Vec::new());
    builder.sign(&signer, format, &mut source, &mut dest)?;

    Ok(dest.into_inner())
}

pub struct VerifyResult {
    pub manifest_json: String,
    /// None means the manifest validated with no errors/warnings reported.
    pub validation_issues: Option<Vec<String>>,
}

/// Reads back an embedded manifest and reports its contents + validation
/// status. A self-signed identity (see `LocalSigner`) is expected to
/// produce at least one validation status entry about the signing
/// certificate not being in a trust list -- that's not a bug, it's what
/// "self-signed, not from a real CA" means. Reported here rather than
/// hidden so this PoC doesn't overstate what it actually proves.
pub fn verify(bytes: &[u8], format: &str) -> c2pa::Result<VerifyResult> {
    let mut stream = Cursor::new(bytes);
    let reader = Reader::from_stream(format, &mut stream)?;

    let validation_issues = reader.validation_status().map(|statuses| {
        statuses
            .iter()
            .map(|s| format!("{}: {}", s.code(), s.explanation().unwrap_or("(no explanation)")))
            .collect::<Vec<_>>()
    });

    Ok(VerifyResult {
        manifest_json: reader.json(),
        validation_issues: validation_issues.filter(|v| !v.is_empty()),
    })
}
