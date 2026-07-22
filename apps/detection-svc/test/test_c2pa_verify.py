import subprocess

import pytest

import c2pa_verify
from c2pa_verify import C2paVerifyResult, verify_c2pa


def test_missing_binary_raises_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("RUST_CORE_BIN", str(tmp_path / "does-not-exist"))

    with pytest.raises(FileNotFoundError):
        verify_c2pa(str(tmp_path / "irrelevant.png"))


class _FakeCompletedProcess:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_no_manifest_returns_none_manifest(monkeypatch, tmp_path):
    fake_bin = tmp_path / "rust-core"
    fake_bin.write_text("")  # just needs to exist for the .exists() check
    monkeypatch.setenv("RUST_CORE_BIN", str(fake_bin))
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeCompletedProcess("[c2pa-verify] no manifest found\n")
    )

    result = verify_c2pa("irrelevant.jpg", "jpg")

    assert result.has_manifest is False
    assert result.ownership is None
    assert result.signed_by_dontai is False
    assert result.validation_issues is None


def test_parses_a_real_manifest_and_reported_issues(monkeypatch, tmp_path):
    fake_bin = tmp_path / "rust-core"
    fake_bin.write_text("")
    monkeypatch.setenv("RUST_CORE_BIN", str(fake_bin))

    stdout = """[c2pa-verify] manifest:
{
  "active_manifest": "urn:c2pa:abc",
  "manifests": {
    "urn:c2pa:abc": {
      "assertions": [
        { "label": "c2pa.actions.v2", "data": {} },
        {
          "label": "com.dontai.ownership",
          "data": { "doNotTrain": true, "title": "t", "creatorId": "c1", "perceptualHash": "0xdead" }
        }
      ],
      "signature_info": { "issuer": "DONTAI" }
    }
  }
}
[c2pa-verify] validation reported 1 issue(s):
  - signingCredential.untrusted: signing certificate untrusted
"""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(stdout))

    result = verify_c2pa("candidate.png", "png")

    assert result.has_manifest is True
    assert result.signed_by_dontai is True
    assert result.ownership == {"doNotTrain": True, "title": "t", "creatorId": "c1", "perceptualHash": "0xdead"}
    assert result.validation_issues == ["signingCredential.untrusted: signing certificate untrusted"]


def test_manifest_from_a_non_dontai_signer_has_no_ownership_and_is_not_signed_by_dontai(monkeypatch, tmp_path):
    """A manifest can exist without ever having gone through this project's
    own protect() pipeline (some other tool, or a bad actor trying to
    launder provenance) -- ownership/signed_by_dontai must not just assume
    "a manifest exists" means "it's ours"."""
    fake_bin = tmp_path / "rust-core"
    fake_bin.write_text("")
    monkeypatch.setenv("RUST_CORE_BIN", str(fake_bin))

    stdout = """[c2pa-verify] manifest:
{
  "active_manifest": "urn:c2pa:xyz",
  "manifests": {
    "urn:c2pa:xyz": {
      "assertions": [],
      "signature_info": { "issuer": "SomeOtherTool" }
    }
  }
}
[c2pa-verify] validation: OK, no issues reported
"""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(stdout))

    result = verify_c2pa("candidate.png", "png")

    assert result.has_manifest is True
    assert result.signed_by_dontai is False
    assert result.ownership is None
    assert result.validation_issues is None


def test_raises_runtime_error_on_nonzero_exit(monkeypatch, tmp_path):
    fake_bin = tmp_path / "rust-core"
    fake_bin.write_text("")
    monkeypatch.setenv("RUST_CORE_BIN", str(fake_bin))
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeCompletedProcess("", returncode=101, stderr="panicked")
    )

    with pytest.raises(RuntimeError):
        verify_c2pa("candidate.png", "png")


def test_ownership_and_signed_by_dontai_are_false_with_no_manifest():
    result = C2paVerifyResult(manifest=None, validation_issues=None)
    assert result.ownership is None
    assert result.signed_by_dontai is False
