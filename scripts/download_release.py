#!/usr/bin/env python3
"""Download a pinned release from the repository's dataset catalog."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", help="Key in datasets/releases.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    releases = json.loads((root / "datasets/releases.json").read_text())
    if args.release not in releases:
        parser.error("Choose one of: " + ", ".join(releases))
    destination = args.output_dir.expanduser().resolve()
    if destination == root or root in destination.parents:
        parser.error("Store generated datasets outside the code repository")
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    load_dotenv(root / ".env")

    release = releases[args.release]
    for entry in release["files"]:
        cached = Path(hf_hub_download(release["repo_id"], entry["path"],
                                     repo_type="dataset", revision=release["revision"]))
        expected = entry.get("sha256")
        if expected and hashlib.sha256(cached.read_bytes()).hexdigest() != expected:
            raise ValueError("Checksum mismatch: " + entry["path"])
        output = destination / args.release / entry["path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if hashlib.sha256(output.read_bytes()).digest() != hashlib.sha256(cached.read_bytes()).digest():
                raise FileExistsError(f"Different data already exists: {output}")
        else:
            shutil.copy2(cached, output)
        print(output)


if __name__ == "__main__":
    main()
