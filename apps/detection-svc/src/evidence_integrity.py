"""Evidence manifest signing, verification, and local write-once sealing."""

from __future__ import annotations

import base64
import json
import os
import hashlib
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical_manifest_bytes(manifest: dict) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class EvidenceSigner:
    def __init__(self, private_key: Ed25519PrivateKey, key_id: str = "local-ed25519") -> None:
        self.private_key = private_key
        self.key_id = key_id

    @classmethod
    def from_pem(cls, path: str | Path, key_id: str = "local-ed25519") -> "EvidenceSigner":
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("evidence signing key must be Ed25519")
        return cls(key, key_id)

    @classmethod
    def generate(cls, path: str | Path, key_id: str = "local-ed25519") -> "EvidenceSigner":
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = Ed25519PrivateKey.generate()
        destination.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        os.chmod(destination, 0o600)
        return cls(key, key_id)

    def sign_manifest(self, path: str | Path) -> dict:
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["signatureStatus"] = "SIGNED"
        signature = self.private_key.sign(canonical_manifest_bytes(manifest))
        public_key = self.private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        manifest["signature"] = {
            "algorithm": "Ed25519", "keyId": self.key_id,
            "publicKey": base64.b64encode(public_key).decode("ascii"),
            "value": base64.b64encode(signature).decode("ascii"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def verify_manifest_signature(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = manifest.get("signature")
    if not signature:
        return {"valid": False, "status": "UNSIGNED"}
    try:
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(signature["publicKey"]))
        public_key.verify(base64.b64decode(signature["value"]), canonical_manifest_bytes(manifest))
        return {"valid": True, "status": "VALID", "keyId": signature.get("keyId")}
    except Exception as exc:
        return {"valid": False, "status": "INVALID", "error": str(exc)}


def verify_evidence_directory(directory: str | Path) -> dict:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"valid": False, "status": "MANIFEST_NOT_FOUND", "files": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_results = []
    all_valid = True
    for item in manifest.get("files", []):
        path = (root / item["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            file_results.append({"path": item["path"], "valid": False, "error": "path escapes evidence directory"})
            all_valid = False
            continue
        if not path.is_file():
            file_results.append({"path": item["path"], "valid": False, "error": "missing"})
            all_valid = False
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        valid = digest == item.get("sha256") and path.stat().st_size == item.get("size")
        file_results.append({"path": item["path"], "valid": valid, "sha256": digest})
        all_valid &= valid
    signature = verify_manifest_signature(manifest_path)
    return {"valid": all_valid and signature["valid"], "status": "VALID" if all_valid and signature["valid"] else "INVALID",
            "signature": signature, "files": file_results, "sealed": (root / ".sealed").exists()}


def seal_directory(directory: str | Path) -> Path:
    """Local write-once emulation. Production must replace this with Object Lock/WORM."""
    root = Path(directory)
    marker = root / ".sealed"
    if marker.exists():
        raise FileExistsError(f"evidence directory is already sealed: {root}")
    marker.write_text("sealed\n", encoding="ascii")
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return marker
