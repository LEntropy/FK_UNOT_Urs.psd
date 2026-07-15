//! Diagnostic tool from investigating the (now-resolved) C2PA
//! `claimSignature.mismatch` issue documented in README.md's "C2PA
//! manifest" section: confirms the certificate's embedded public key
//! matches the key actually used for signing, and that raw ed25519-dalek
//! sign/verify works correctly in isolation -- ruling out a key-mismatch or
//! basic-crypto bug as the cause (the real cause was a missing Organization
//! subject attribute, a known upstream `c2pa` crate bug). Kept as evidence
//! for that investigation, not part of the normal build.

use ed25519_dalek::pkcs8::DecodePrivateKey;
use ed25519_dalek::SigningKey;
use x509_parser::prelude::{FromDer, X509Certificate};
use x509_parser::public_key::PublicKey;

fn main() {
    let rcgen_keypair = rcgen::KeyPair::generate_for(&rcgen::PKCS_ED25519).unwrap();

    let mut params = rcgen::CertificateParams::new(vec!["dontai.local".to_string()]).unwrap();
    params.is_ca = rcgen::IsCa::ExplicitNoCa;
    params.key_usages = vec![rcgen::KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::EmailProtection];
    params.use_authority_key_identifier_extension = true;

    let cert = params.self_signed(&rcgen_keypair).unwrap();

    let signing_key = SigningKey::from_pkcs8_der(rcgen_keypair.serialized_der()).unwrap();
    let derived_pubkey = signing_key.verifying_key().to_bytes();

    let cert_der = cert.der();
    let (_, x509) = X509Certificate::from_der(cert_der).unwrap();
    let spki = x509.public_key();
    let parsed = spki.parsed().unwrap();
    let PublicKey::Unknown(cert_pubkey_bytes) = parsed else {
        panic!("unexpected public key type in cert");
    };

    println!("derived pubkey (from signing key): {}", hex::encode(derived_pubkey));
    println!("cert embedded pubkey:               {}", hex::encode(cert_pubkey_bytes));
    println!("match: {}", derived_pubkey.as_slice() == cert_pubkey_bytes);

    // Also sign+verify directly with ed25519-dalek itself, bypassing c2pa entirely.
    use ed25519_dalek::Signer;
    let msg = b"hello dontai";
    let sig = signing_key.sign(msg);
    let verify_result = signing_key.verifying_key().verify_strict(msg, &sig);
    println!("direct ed25519-dalek self-verify: {:?}", verify_result.is_ok());
}
