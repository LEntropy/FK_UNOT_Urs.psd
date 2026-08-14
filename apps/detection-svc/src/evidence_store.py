"""Evidence storage adapters.

LocalWormEvidenceStore is a development-grade write-once emulator. It gives
the detection pipeline stable semantics (unique object IDs, retention
metadata, overwrite refusal, hash verification and an append-only audit
trail) while keeping the backend replaceable with S3/GCS Object Lock later.
OS administrators can still override local permissions, so this is not a
regulatory WORM substitute.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return safe[:120] or "evidence"


@dataclass(frozen=True)
class StoredEvidence:
    object_id: str
    uri: str
    retained_until: str
    manifest_sha256: str


class LocalWormEvidenceStore:
    backend = "local_worm"

    def __init__(self, root: str | Path, retention_days: int = 365) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        self.root = Path(root).resolve()
        self.retention_days = retention_days
        self.objects = self.root / "objects"
        self.audit_path = self.root / "audit.jsonl"
        self.objects.mkdir(parents=True, exist_ok=True)

    def put(self, source_directory: str | Path, *, case_id: str, evidence_id: str) -> StoredEvidence:
        source = Path(source_directory).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)

        object_id = f"{_safe_component(case_id)}/{_safe_component(evidence_id)}"
        destination = self.objects / object_id
        if destination.exists():
            raise FileExistsError(f"WORM object already exists: {object_id}")

        staging = destination.with_name(destination.name + f".staging-{uuid.uuid4().hex}")
        staging.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(source, staging)
            created_at = datetime.now(timezone.utc)
            retained_until = created_at + timedelta(days=self.retention_days)
            files = []
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                relative = path.relative_to(staging).as_posix()
                files.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})

            worm_manifest = {
                "schemaVersion": 1,
                "backend": self.backend,
                "objectId": object_id,
                "caseId": case_id,
                "evidenceId": evidence_id,
                "createdAt": created_at.isoformat(),
                "retentionMode": "COMPLIANCE_EMULATION",
                "retainedUntil": retained_until.isoformat(),
                "files": files,
            }
            manifest_path = staging / "worm-manifest.json"
            manifest_path.write_text(json.dumps(worm_manifest, indent=2), encoding="utf-8")
            manifest_digest = _sha256(manifest_path)

            # Rename within the same filesystem is atomic. A competing writer
            # cannot silently replace an existing object.
            if destination.exists():
                raise FileExistsError(f"WORM object already exists: {object_id}")
            staging.rename(destination)
            self._seal(destination)
            self._append_audit("PUT", object_id, manifest_digest, retained_until.isoformat())
            return StoredEvidence(object_id, str(destination), retained_until.isoformat(), manifest_digest)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def verify(self, object_id: str) -> dict:
        directory = (self.objects / object_id).resolve()
        try:
            directory.relative_to(self.objects.resolve())
        except ValueError:
            return {"valid": False, "status": "INVALID_OBJECT_ID", "objectId": object_id}
        manifest_path = directory / "worm-manifest.json"
        if not manifest_path.is_file():
            return {"valid": False, "status": "MANIFEST_NOT_FOUND", "objectId": object_id}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = []
        valid = True
        for item in manifest.get("files", []):
            path = (directory / item["path"]).resolve()
            try:
                path.relative_to(directory)
            except ValueError:
                results.append({"path": item["path"], "valid": False, "error": "path escape"})
                valid = False
                continue
            item_valid = path.is_file() and path.stat().st_size == item["size"] and _sha256(path) == item["sha256"]
            results.append({"path": item["path"], "valid": item_valid})
            valid &= item_valid
        return {
            "valid": valid,
            "status": "VALID" if valid else "INVALID",
            "objectId": object_id,
            "retainedUntil": manifest.get("retainedUntil"),
            "files": results,
            "sealed": (directory / ".worm-sealed").exists(),
        }

    def _seal(self, directory: Path) -> None:
        marker = directory / ".worm-sealed"
        marker.write_text("local WORM emulation; see worm-manifest.json\n", encoding="utf-8")
        for path in directory.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        directory.chmod(0o555)

    def _append_audit(self, action: str, object_id: str, manifest_sha256: str, retained_until: str) -> None:
        entry = {
            "timestamp": time.time(),
            "action": action,
            "objectId": object_id,
            "manifestSha256": manifest_sha256,
            "retainedUntil": retained_until,
        }
        fd = os.open(self.audit_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o444)
        try:
            os.write(fd, (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
