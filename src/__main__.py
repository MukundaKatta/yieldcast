"""CLI for yieldcast."""
import sys, json, argparse
from .core import Yieldcast

def main():
    parser = argparse.ArgumentParser(description="YieldCast — AI Crop Yield Predictor. Predict crop yields using weather, soil, and satellite data.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Yieldcast()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.track(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"yieldcast v0.1.0 — YieldCast — AI Crop Yield Predictor. Predict crop yields using weather, soil, and satellite data.")

if __name__ == "__main__":
    main()
