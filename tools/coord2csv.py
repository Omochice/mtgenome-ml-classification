import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert json that has coordinates to CSV"
    )
    parser.add_argument("json", nargs=1, help="Source json file")

    args = parser.parse_args()

    for file in args.json:
        with open(file) as f:
            d = json.load(f)

        for row in d:
            print(",".join(map(str, row)))
