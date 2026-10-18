from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import LINE_SPECS, PROFILES, run
from .holling import DOMAIN_PRESETS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HP/Holling ecology NDTL experiments."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--out-root", default="outputs/hp")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Reuse a matching CAE instead of training a new one.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--line", choices=["both", *sorted(LINE_SPECS)], default="both")
    parser.add_argument(
        "--domain-mode",
        choices=[*sorted(DOMAIN_PRESETS), "trajectory_union"],
        default="fixed02",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = run(
        profile_name=args.profile,
        out_root=Path(args.out_root),
        device_arg=args.device,
        seed=args.seed,
        line=args.line,
        domain_mode=args.domain_mode,
        checkpoint=args.checkpoint,
    )
    print(
        f"completed profile={args.profile}, line={args.line}, "
        f"domain_mode={args.domain_mode}, lines={list(report.get('lines', {}).keys())}"
    )


if __name__ == "__main__":
    main()
