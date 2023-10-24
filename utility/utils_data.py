import os.path as osp
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

    return data
