from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
CHECKSUM_FILE = ROOT / "SHA256SUMS.txt"
PACKAGE = ROOT / "dqn_family_satellite_ground"
LOCKED_RESULTS = PACKAGE / "results" / "reviewer_revision_state_complete"
REVISION_RESULTS = PACKAGE / "results" / "revision_round2"
VARIANTS = {
    "standard_dqn",
    "full_action_q_dqn",
    "immediate_advantage_dqn",
    "centered_full_action_dqn",
    "double_dqn",
    "double_centered_full_action_dqn",
}
EXPECTED_LOCKED_COUNTS = {
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


def package_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]


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
        for path in package_files() if path != CHECKSUM_FILE
    }
    if set(expected) != actual:
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise AssertionError(f"inventory mismatch; missing={missing}; extra={extra}")
    mismatches = [
        relative for relative, digest in expected.items()
        if sha256(ROOT / relative) != digest
    ]
    if mismatches:
        raise AssertionError(f"SHA-256 mismatch: {mismatches}")


def verify_scope() -> None:
    files = package_files()
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    latex = [path for path in files if path.suffix.lower() == ".tex"]
    if images or latex:
        raise AssertionError(f"excluded manuscript assets found: {images + latex}")
    pdfs = [path for path in files if path.suffix.lower() == ".pdf"]
    if pdfs != [ROOT / "dqn_family_satellite_ground_manuscript.pdf"]:
        raise AssertionError(f"expected exactly one manuscript PDF, found {pdfs}")
    forbidden_directory_names = {"archive", "review", "figures", "paper"}
    forbidden = [
        path for path in ROOT.rglob("*")
        if path.is_dir() and ".git" not in path.relative_to(ROOT).parts
        and path.name.lower() in forbidden_directory_names
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


def verify_locked_checkpoints() -> None:
    checkpoints = sorted(LOCKED_RESULTS.rglob("*.pth"))
    curves = sorted(LOCKED_RESULTS.rglob("*__curve.csv"))
    if len(checkpoints) != 180 or len(curves) != 180:
        raise AssertionError(
            f"expected 180 locked checkpoints and curves, found {len(checkpoints)} and {len(curves)}"
        )
    counts: Counter[str] = Counter()
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        variant = payload["variant"]
        counts[variant] += 1
        if variant not in EXPECTED_LOCKED_COUNTS:
            raise AssertionError(f"unexpected variant {variant!r} in {path.name}")
        if int(payload["training_steps"]) != 38_400:
            raise AssertionError(f"training-budget mismatch in {path.name}")
        if int(payload["model_seed"]) not in range(800, 820):
            raise AssertionError(f"model-seed mismatch in {path.name}")
    if dict(counts) != EXPECTED_LOCKED_COUNTS:
        raise AssertionError(f"locked checkpoint counts differ: {dict(counts)}")
    with (LOCKED_RESULTS / "model_manifest.csv").open(encoding="utf-8", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    with (LOCKED_RESULTS / "preview_mismatch_model_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        mismatch_rows = list(csv.DictReader(handle))
    if len(main_rows) != 140 or len(mismatch_rows) != 40:
        raise AssertionError(
            f"locked manifest counts differ: main={len(main_rows)}, mismatch={len(mismatch_rows)}"
        )


def verify_revision_checkpoints() -> None:
    gamma = sorted((REVISION_RESULTS / "gamma_models").glob("*.pth"))
    gamma_curves = sorted((REVISION_RESULTS / "gamma_models").glob("*__curve.csv"))
    length = sorted((REVISION_RESULTS / "training_length_models").glob("*.pth"))
    length_curves = sorted((REVISION_RESULTS / "training_length_models").glob("*__curve.csv"))
    if (len(gamma), len(gamma_curves), len(length), len(length_curves)) != (480, 480, 360, 360):
        raise AssertionError(
            "revision checkpoint counts differ: "
            f"gamma={len(gamma)}/{len(gamma_curves)}, length={len(length)}/{len(length_curves)}"
        )
    gamma_counts: Counter[str] = Counter()
    for path in gamma:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        gamma_counts[payload["variant"]] += 1
        if payload["variant"] not in VARIANTS or int(payload["training_steps"]) != 38_400:
            raise AssertionError(f"invalid gamma checkpoint: {path.name}")
        if int(payload["model_seed"]) not in range(800, 820):
            raise AssertionError(f"model-seed mismatch in {path.name}")
    if gamma_counts != Counter({variant: 80 for variant in VARIANTS}):
        raise AssertionError(f"gamma variant counts differ: {dict(gamma_counts)}")
    length_counts: Counter[str] = Counter()
    for path in length:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        length_counts[payload["variant"]] += 1
        checkpoint = int(payload["checkpoint_episode"])
        if payload["variant"] not in VARIANTS or checkpoint not in {300, 600, 900}:
            raise AssertionError(f"invalid training-length checkpoint: {path.name}")
        if int(payload["training_steps"]) != checkpoint * 64:
            raise AssertionError(f"training-length budget mismatch in {path.name}")
    if length_counts != Counter({variant: 60 for variant in VARIANTS}):
        raise AssertionError(f"training-length variant counts differ: {dict(length_counts)}")


def verify_revision_tables() -> None:
    for value in ("0p9", "0p95", "0p97", "0p99"):
        manifest = pd.read_csv(REVISION_RESULTS / f"gamma_{value}_manifest.csv")
        evaluation = pd.read_csv(REVISION_RESULTS / f"gamma_{value}_evaluation.csv")
        if len(manifest) != 120 or len(evaluation) != 2400:
            raise AssertionError(f"gamma {value} row counts differ")
    if len(pd.read_csv(REVISION_RESULTS / "training_length_manifest.csv")) != 360:
        raise AssertionError("training-length manifest row count differs")
    if len(pd.read_csv(REVISION_RESULTS / "training_length_evaluation.csv")) != 7200:
        raise AssertionError("training-length evaluation row count differs")
    if len(pd.read_csv(REVISION_RESULTS / "baseline_evaluation.csv")) != 2000:
        raise AssertionError("baseline evaluation row count differs")
    validation = json.loads((REVISION_RESULTS / "revision_validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or not validation.get(
        "checkpoint_600_matches_locked_reference", False
    ):
        raise AssertionError("revision validation record is incomplete")


def main() -> None:
    verify_inventory_and_hashes()
    verify_scope()
    verify_locked_checkpoints()
    verify_revision_checkpoints()
    verify_revision_tables()
    print("package_verification=passed")
    print("files_verified=", len(read_checksums()))
    print("checkpoints_verified=1020")
    print("training_curves_verified=1020")
    print("article_images_in_package=0")


if __name__ == "__main__":
    main()
