import json
import os.path as osp
import time

import pandas as pd



def load_data(args, subset: str = None):
    """
    Load the arXiv metadata.
    Args:
        args: Arguments. We need the data_dir attribute.
        subset: str: Subset of the arXiv metadata to load.

    Returns:
        data (pd.DataFrame): DataFrame containing the arXiv metadata.
    """

    t0 = time.time()
    if subset is None:
        data = pd.read_pickle(osp.join(args.data_dir, "arXiv_metadata.pkl"))
    else:
        if subset == "first_100":
            path = osp.join(args.data_dir, "arXiv_metadata_first_100_entries.xlsx")

        elif subset == "last_100":
            path = osp.join(args.data_dir, "arXiv_metadata_last_100_entries.xlsx")

        elif subset == "last_10000":
            path = osp.join(args.data_dir, "arXiv_metadata_last_10000_entries.xlsx")

        else:
            raise ValueError(f"subset {subset} not recognized")

        data = pd.read_excel(path)

    print(f"Loaded {len(data)} entries in {(time.time() - t0):.3f} secs.")
    return data

def load_tag2papers(args):
    tag2papers = json.load(open(osp.join(args.data_dir, "tag2papers.json"), "r"))
    return tag2papers


