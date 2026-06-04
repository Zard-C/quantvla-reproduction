"""Download a small LeRobot dataset subset for real-data GR00T validation.

The LIBERO LeRobot datasets are large. This helper downloads only metadata and
the first few episode parquet/video files needed for an offline validation set.
It also installs the GR00T LIBERO modality mapping when requested.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def episode_name(index: int) -> str:
    return f"episode_{index:06d}"


def build_allow_patterns(start_episode: int, num_episodes: int) -> list[str]:
    patterns = ["meta/*"]
    for episode in range(start_episode, start_episode + num_episodes):
        chunk = episode // 1000
        name = episode_name(episode)
        patterns.extend(
            [
                f"data/chunk-{chunk:03d}/{name}.parquet",
                f"videos/chunk-{chunk:03d}/observation.images.image/{name}.mp4",
                f"videos/chunk-{chunk:03d}/observation.images.wrist_image/{name}.mp4",
            ]
        )
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--modality-json",
        type=Path,
        help="Optional GR00T LIBERO modality.json to copy into local-dir/meta/.",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        import os

        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    from huggingface_hub import snapshot_download

    allow_patterns = build_allow_patterns(args.start_episode, args.num_episodes)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        token=args.token,
    )

    if args.modality_json is not None:
        target = args.local_dir / "meta" / "modality.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.modality_json, target)

    print(f"Downloaded subset to {path}")
    print("Allow patterns:")
    for pattern in allow_patterns:
        print(f"  {pattern}")


if __name__ == "__main__":
    main()
