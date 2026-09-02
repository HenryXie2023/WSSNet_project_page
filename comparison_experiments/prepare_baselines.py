from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = Path(__file__).with_name("baseline_sources.yaml")


def load_sources(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("baselines", {})


def repo_dir_name(repository: str) -> str:
    name = Path(urlparse(repository).path).name
    return name[:-4] if name.endswith(".git") else name


def resolve_repo_path(entry: dict, third_party_root: Path) -> Path:
    configured = Path(entry["local_repository_path"])
    if configured.is_absolute():
        return configured
    if configured.parts and configured.parts[0] == "..":
        return (ROOT / configured).resolve()
    return (third_party_root / configured).resolve()


def clone_repository(repository: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repository, str(destination)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate or download external baseline repositories outside this release.")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument("--third-party-root", default="../third_party_baselines")
    parser.add_argument("--download", action="store_true", help="Clone missing repositories into --third-party-root.")
    args = parser.parse_args()

    third_party_root = (ROOT / args.third_party_root).resolve()
    baselines = load_sources(Path(args.sources))
    if not baselines:
        raise ValueError("No baselines found in baseline_sources.yaml")

    for key, entry in baselines.items():
        destination = resolve_repo_path(entry, third_party_root)
        if not destination.name:
            destination = third_party_root / repo_dir_name(entry["repository"])
        if destination.exists():
            status = "found"
        elif args.download:
            clone_repository(entry["repository"], destination)
            status = "downloaded"
        else:
            status = "missing"
        print(f"{key}: {status}: {destination}")


if __name__ == "__main__":
    main()
