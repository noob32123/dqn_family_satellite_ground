from __future__ import annotations

from collections import Counter
import csv
import hashlib
from pathlib import Path
import re

import torch


ROOT = Path(__file__).resolve().parent
CHECKSUM_FILE = ROOT / "SHA256SUMS.txt"
RESULTS = (
    ROOT
    / "dqn_family_satellite_ground"
    / "results"
    / "reviewer_revision_state_complete"
)
EXPECTED_VARIANT_COUNTS = {
    "standard_dqn": 20,
    "full_action_q_dqn": 20,
    "immediate_advantage_dqn": 20,
    "centered_full_action_dqn": 60,
    "double_dqn": 20,
    "double_centered_full_action_dqn": 20,
    "contextual_bandit": 20,
}
TEXT_SUFFIXES = {".py", ".m", ".md", ".txt", ".csv", ".json", ".tsv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".tif", ".tiff", ".eps"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksums() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    return entries


def verify_inventory_and_hashes() -> None:
    expected = read_checksums()
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != CHECKSUM_FILE
    }
    if set(expected) != actual:
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise AssertionError(f"inventory mismatch; missing={missing}; extra={extra}")
    mismatches = [
        relative
        for relative, digest in expected.items()
        if sha256(ROOT / relative) != digest
    ]
    if mismatches:
        raise AssertionError(f"SHA-256 mismatch: {mismatches}")


def verify_scope() -> None:
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    latex = [path for path in files if path.suffix.lower() == ".tex"]
    if images or latex:
        raise AssertionError(f"excluded manuscript assets found: {images + latex}")
    pdfs = [path for path in files if path.suffix.lower() == ".pdf"]
    if pdfs != [ROOT / "dqn_family_satellite_ground_manuscript.pdf"]:
        raise AssertionError(f"expected exactly one manuscript PDF, found {pdfs}")
    forbidden_directory_names = {"archive", "review", "figures", "paper"}
    forbidden = [
        path
        for path in ROOT.rglob("*")
        if path.is_dir() and path.name.lower() in forbidden_directory_names
    ]
    if forbidden:
        raise AssertionError(f"excluded directories found: {forbidden}")

    legacy_token = "c" + "bad"
    legacy_namespace = "md_" + legacy_token + "_dqn"
    semantic_legacy = re.compile(
        rf"(?<![0-9a-f])(?:{re.escape(legacy_namespace)}|{re.escape(legacy_token)})(?![0-9a-f])"
    )
    for path in files:
        relative = path.relative_to(ROOT).as_posix().lower()
        if semantic_legacy.search(relative):
            raise AssertionError(f"legacy identifier in path: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="strict").lower()
            if semantic_legacy.search(text):
                raise AssertionError(f"legacy identifier in text file: {relative}")


def verify_checkpoints() -> None:
    checkpoints = sorted(RESULTS.rglob("*.pth"))
    curves = sorted(RESULTS.rglob("*__curve.csv"))
    if len(checkpoints) != 180 or len(curves) != 180:
        raise AssertionError(
            f"expected 180 checkpoints and 180 curves, found {len(checkpoints)} and {len(curves)}"
        )
    counts: Counter[str] = Counter()
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        variant = payload["variant"]
        counts[variant] += 1
        if variant not in EXPECTED_VARIANT_COUNTS:
            raise AssertionError(f"unexpected variant {variant!r} in {path.name}")
        if int(payload["training_steps"]) != 38_400:
            raise AssertionError(f"training-budget mismatch in {path.name}")
        if int(payload["model_seed"]) not in range(800, 820):
            raise AssertionError(f"model-seed mismatch in {path.name}")
    if dict(counts) != EXPECTED_VARIANT_COUNTS:
        raise AssertionError(f"checkpoint variant counts differ: {dict(counts)}")

    with (RESULTS / "model_manifest.csv").open(encoding="utf-8", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    with (RESULTS / "preview_mismatch_model_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        mismatch_rows = list(csv.DictReader(handle))
    if len(main_rows) != 140 or len(mismatch_rows) != 40:
        raise AssertionError(
            f"manifest counts differ: main={len(main_rows)}, mismatch={len(mismatch_rows)}"
        )


def main() -> None:
    verify_inventory_and_hashes()
    verify_scope()
    verify_checkpoints()
    print("package_verification=passed")
    print("files_verified=", len(read_checksums()))
    print("checkpoints_verified=180")
    print("training_curves_verified=180")
    print("article_images_in_package=0")


if __name__ == "__main__":
    main()
