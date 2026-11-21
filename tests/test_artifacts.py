"""Web artifact schema validation and `mlfte ingest` on a synthetic fixture zip."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from mlfte.export.ingest import ingest
from mlfte.export.web_artifacts import ARTIFACT_FILES, COMMON_FIELDS, validate_payload


def _payload(name: str, profile: str = "cpu", device: str = "cpu") -> dict:
    base = {
        "schema_version": 1,
        "profile": profile,
        "hardware": {"device": device, "cpu_count_logical": 2},
        "git_commit": "deadbeef",
        "data_fingerprint": "abc123",
    }
    if name == "summary.json":
        base["headline"] = {"primary_metric": "execution_match", "test_n": 10}
    if name == "examples.json":
        base["examples"] = [{"id": "x", "category": "fixed"}]
    if name == "comparison.json":
        base["runs"] = {}
    if name == "env.json":
        base["env"] = {
            "hardware": {
                "device": device,
                "gpu_available": device == "cuda",
                "gpu_name": "T4" if device == "cuda" else None,
            }
        }
    return base


@pytest.mark.parametrize("name", ARTIFACT_FILES)
def test_valid_payload_passes(name):
    validate_payload(name, _payload(name))


def test_missing_common_field_fails():
    payload = _payload("summary.json")
    del payload["profile"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_payload("summary.json", payload)


def test_wrong_schema_version_fails():
    payload = _payload("summary.json")
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        validate_payload("summary.json", payload)


def test_examples_must_be_a_list():
    payload = _payload("examples.json")
    payload["examples"] = {}
    with pytest.raises(ValueError, match="examples"):
        validate_payload("examples.json", payload)


def test_non_object_payload_fails():
    with pytest.raises(ValueError):
        validate_payload("summary.json", [])


def _make_zip(tmp_path: Path, profile: str = "cpu", device: str = "cpu") -> Path:
    export = tmp_path / "bundle" / "results" / "export"
    export.mkdir(parents=True)
    for name in ARTIFACT_FILES:
        (export / name).write_text(json.dumps(_payload(name, profile, device)))
    zip_path = tmp_path / "results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in export.rglob("*"):
            zf.write(path, path.relative_to(tmp_path / "bundle"))
    return zip_path


def test_ingest_installs_validated_artifacts(tmp_path):
    zip_path = _make_zip(tmp_path)
    web = tmp_path / "web"
    info = ingest(zip_path, expect_profile="cpu", web_data_dir=web)
    assert info["profile"] == "cpu"
    for name in ARTIFACT_FILES:
        assert (web / name).exists()
        validate_payload(name, json.loads((web / name).read_text()))


def test_ingest_rejects_profile_mismatch(tmp_path):
    zip_path = _make_zip(tmp_path)
    with pytest.raises(ValueError, match="profile"):
        ingest(zip_path, expect_profile="gpu", web_data_dir=tmp_path / "web")


def test_ingest_rejects_gpu_claim_without_gpu(tmp_path):
    zip_path = _make_zip(tmp_path, profile="gpu", device="cuda")
    export_env = tmp_path / "bundle" / "results" / "export" / "env.json"
    payload = json.loads(export_env.read_text())
    payload["env"]["hardware"]["gpu_available"] = False
    export_env.write_text(json.dumps(payload))
    zip_path.unlink()
    zip_path = tmp_path / "results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in (tmp_path / "bundle").rglob("*"):
            zf.write(path, path.relative_to(tmp_path / "bundle"))
    with pytest.raises(ValueError, match="no GPU"):
        ingest(zip_path, expect_profile="gpu", web_data_dir=tmp_path / "web")


def test_ingest_rejects_zip_without_artifacts(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(FileNotFoundError):
        ingest(zip_path, expect_profile="cpu", web_data_dir=tmp_path / "web")


def test_common_fields_are_the_documented_contract():
    assert set(COMMON_FIELDS) == {"schema_version", "profile", "hardware", "git_commit"}
