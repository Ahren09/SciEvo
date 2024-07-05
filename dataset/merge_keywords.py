"""
Date: 2024.6.26

Merge keywords from different years into one file.

"""
import json
import os
import sys

from tqdm import trange

from utility.utils_misc import project_setup

sys.path.append(os.path.abspath('.'))
from arguments import parse_args


def main():
    project_setup()
    args = parse_args()
    all_keywords = {}

    for start_year in trange(args.start_year, args.end_year):
        path = os.path.join(args.data_dir, "NLP", "arXiv", f"{args.feature_name}_keywords", f"{args.feature_name}_keywords_{start_year}.json")
        with open(path, 'r') as f:
            data = json.load(f)
            if not data:
                continue

            all_keywords.update(data)

    path = os.path.join(args.data_dir, "NLP", "arXiv", f"{args.feature_name}_keywords.json")

    with open(path, 'w') as f:
        json.dump(all_keywords, f)

    print("Done!")


if __name__ == "__main__":
    main()
