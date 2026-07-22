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
use std::path::Path;

use c2pa::{Builder, Reader, Signer, SigningAlg};
use ed25519_dalek::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use ed25519_dalek::{Signer as _, SigningKey};
use rustls_pki_types::PrivatePkcs8KeyDer;

/// A self-signed Ed25519 identity for signing C2PA manifests locally.
/// Not a substitute for a real CA-issued signing certificate -- see module docs.
pub struct LocalSigner {
    signing_key: SigningKey,
    cert_der: Vec<u8>,
}

impl LocalSigner {
    /// Generates a fresh self-signed Ed25519 identity, discarding the key
    /// material immediately after this call returns. Every call produces a
    /// *different* keypair -- fine for the PoC's own sign/verify-within-
    /// one-process tests, but wrong for a real pipeline: every image would
    /// get signed by a different, unrelated "identity," so there'd be no
    /// way to say "these two images were both signed by DONTAI." Real
    /// pipeline callers should use `load_or_generate` instead.
    pub fn generate() -> Self {
        let rcgen_keypair =
            rcgen::KeyPair::generate_for(&rcgen::PKCS_ED25519).expect("failed to generate Ed25519 keypair");
        // Re-derive the ed25519-dalek signing key from the same PKCS8 DER
        // rcgen generated, so the certificate's public key and the key we
        // actually sign with are guaranteed to match.
        let signing_key =
            SigningKey::from_pkcs8_der(rcgen_keypair.serialized_der()).expect("failed to parse Ed25519 PKCS8 key");
        let cert_der = Self::self_sign_cert(&rcgen_keypair);
        Self { signing_key, cert_der }
    }

    /// Loads a persisted Ed25519 keypair (PKCS8 DER) from `key_path` if one
    /// exists, or generates and persists a fresh one there. Same
    /// public/private keypair on every call this way -- the fix for
    /// `generate()`'s "different identity per call" problem above. The
    /// self-signed *certificate* wrapping the key is still rebuilt fresh
    /// every call (a new serial number and validity window each time,
    /// exactly like reissuing a real cert around the same key would) --
    /// what makes the identity recognizably "the same DONTAI" across
    /// images is the public key the cert carries, not needing the cert
    /// bytes themselves to be byte-identical too.
    ///
    /// Best-effort persistence: if the file can't be read or written (odd
    /// permissions, disk full, whatever), this falls back to `generate()`'s
    /// behavior rather than failing the whole sign operation -- a C2PA
    /// manifest signed by a fresh throwaway identity is still a real,
    /// validly-signed manifest, just not one that ties back to a stable
    /// DONTAI identity across images. Logged to stderr either way so this
    /// degradation is visible, not silent.
    pub fn load_or_generate(key_path: &Path) -> Self {
        if let Ok(der_bytes) = std::fs::read(key_path) {
            match SigningKey::from_pkcs8_der(&der_bytes) {
                Ok(signing_key) => {
                    let rcgen_keypair = rcgen::KeyPair::from_pkcs8_der_and_sign_algo(
                        &PrivatePkcs8KeyDer::from(der_bytes.as_slice()),
                        &rcgen::PKCS_ED25519,
                    )
                    .expect("persisted C2PA signing key's DER bytes parsed by ed25519-dalek but rejected by rcgen -- should be impossible for a key this same module wrote");
                    let cert_der = Self::self_sign_cert(&rcgen_keypair);
                    return Self { signing_key, cert_der };
                }
                Err(err) => {
                    eprintln!(
                        "[c2pa] existing signing key at {} could not be parsed ({err}), generating a new one",
                        key_path.display()
                    );
                }
            }
        }

        let fresh = Self::generate();
        if let Some(parent) = key_path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                eprintln!("[c2pa] could not create {} to persist the signing key ({err}) -- this identity will not survive past this process", parent.display());
                return fresh;
            }
        }
        match fresh.signing_key.to_pkcs8_der() {
            Ok(der) => {
                if let Err(err) = std::fs::write(key_path, der.as_bytes()) {
                    eprintln!("[c2pa] could not persist signing key to {} ({err}) -- this identity will not survive past this process", key_path.display());
                }
            }
            Err(err) => {
                eprintln!("[c2pa] could not encode signing key for persistence ({err}) -- this identity will not survive past this process");
            }
        }
        fresh
    }

    /// Builds the X.509v3 self-signed certificate `c2pa`'s signing/
    /// verification path requires around a given keypair. Shared by
    /// `generate()` and `load_or_generate()` so both produce a cert with
    /// identical required extensions, differing only in which keypair (a
    /// fresh one, or a loaded one) it wraps.
    /// The Ed25519 public key this identity signs with, hex-encoded. Same
    /// value across every call sharing a `load_or_generate` key path --
    /// useful for a caller (or a test) that wants to confirm "these two
    /// signed images really do share the same DONTAI identity" without
    /// needing to parse the X.509 cert bytes to get there.
    pub fn public_key_hex(&self) -> String {
        self.signing_key.verifying_key().to_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    fn self_sign_cert(rcgen_keypair: &rcgen::KeyPair) -> Vec<u8> {
        let mut params =
            rcgen::CertificateParams::new(vec!["dontai-protection-svc.local".to_string()])
                .expect("failed to build cert params");
        params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "DONTAI Protection Service (self-signed, PoC only)");
        // Required, not decorative: c2pa-rs's cert-profile check rejects
        // certs with no Organization (O) attribute, but instead of
        // surfacing that as a certificate-profile error it gets misreported
        // as claimSignature.mismatch on read-back -- a cryptographically
        // valid signature reads as "invalid" with no hint the real problem
        // is the missing O field. Confirmed as a known upstream bug
        // (github.com/contentauth/c2pa-rs issue #2262) with this exact fix
        // as the reporter's confirmed workaround.
        params
            .distinguished_name
            .push(rcgen::DnType::OrganizationName, "DONTAI");

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
            .self_signed(rcgen_keypair)
            .expect("failed to self-sign certificate");
        cert.der().to_vec()
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
/// caller-supplied JSON -- the real pipeline (orchestrate.py) puts
/// doNotTrain/title/creatorId/perceptualHash here; blockchain-svc's
/// on-chain contentHash/txHash isn't available yet at this point in the
/// pipeline, since on-chain registration is a separate, later step in
/// asset-service's own job state machine, not something protect() itself
/// triggers) and embeds it into `input_bytes`, returning the signed output
/// bytes. `signing_key_path` is forwarded to `LocalSigner::load_or_generate`
/// so every call from the same deployment signs with the same identity.
pub fn sign_and_embed(
    input_bytes: &[u8],
    format: &str,
    title: &str,
    custom_assertion_label: &str,
    custom_assertion: &serde_json::Value,
    signing_key_path: &std::path::Path,
) -> c2pa::Result<Vec<u8>> {
    let manifest_json = serde_json::json!({
        "title": title,
        "claim_generator_info": [{ "name": "dontai-protection-svc/rust-core", "version": env!("CARGO_PKG_VERSION") }],
        "assertions": [
            { "label": "c2pa.actions", "data": { "actions": [{ "action": "c2pa.created" }] } },
        ],
    });

    let mut builder = Builder::from_json(&manifest_json.to_string())?;
    builder.add_assertion_json(custom_assertion_label, custom_assertion)?;

    let signer = LocalSigner::load_or_generate(signing_key_path);

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

/// Distinguishes "there's genuinely no C2PA manifest embedded in this file"
/// (`c2pa::Error::JumbfNotFound`) from every other, real failure (corrupt
/// stream, unsupported format, etc.). Matters a lot for a caller like
/// detection-svc scanning arbitrary URLs found in the wild: the *expected,
/// common* case is "no manifest at all" (never had one, or a
/// redistribution stripped it) -- not an error condition that should look
/// the same as "something is actually broken." Real bug found live: the
/// CLI used to `.expect()` this straight into a panic (exit code 101, a
/// stack trace on stderr) for the ordinary "no manifest" case, making
/// detection-svc's own scan indistinguishable from a real crash without
/// fragile stderr-string-matching.
pub enum VerifyOutcome {
    Found(VerifyResult),
    NoManifest,
}

/// Reads back an embedded manifest and reports its contents + validation
/// status. A self-signed identity (see `LocalSigner`) is expected to
/// produce at least one validation status entry about the signing
/// certificate not being in a trust list -- that's not a bug, it's what
/// "self-signed, not from a real CA" means. Reported here rather than
/// hidden so this PoC doesn't overstate what it actually proves.
pub fn verify(bytes: &[u8], format: &str) -> c2pa::Result<VerifyOutcome> {
    let mut stream = Cursor::new(bytes);
    let reader = match Reader::from_stream(format, &mut stream) {
        Ok(reader) => reader,
        Err(c2pa::Error::JumbfNotFound) => return Ok(VerifyOutcome::NoManifest),
        Err(err) => return Err(err),
    };

    let validation_issues = reader.validation_status().map(|statuses| {
        statuses
            .iter()
            .map(|s| format!("{}: {}", s.code(), s.explanation().unwrap_or("(no explanation)")))
            .collect::<Vec<_>>()
    });

    Ok(VerifyOutcome::Found(VerifyResult {
        manifest_json: reader.json(),
        validation_issues: validation_issues.filter(|v| !v.is_empty()),
    }))
}
