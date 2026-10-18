"""Train and evaluate the 20- or 52-system classification benchmark."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--systems", type=int, choices=[20, 52], default=52)
    args, remaining = parser.parse_known_args()
    if args.systems == 20:
        from .train20 import main as benchmark
    else:
        from .train52 import main as benchmark
    return benchmark(remaining)


if __name__ == "__main__":
    main()
