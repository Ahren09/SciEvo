import os
import time
import traceback

import feedparser

from utility.utils_data import load_arXiv_data, get_arXiv_IDs_of_existing_papers


def download_arXiv_paper_pdf(paper_id):
    """
    Downloads a PDF file of an arXiv paper given its unique paper ID.

    Args:
        paper_id (str): The unique identifier of the arXiv paper.

    Returns:
        None

    This function fetches the PDF file of the arXiv paper with the specified ID
    and saves it to the 'data/pdf' directory.

    Example:
        >>> download_arXiv_paper_pdf("2310.13132")
    """
    paper_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    paper_id = paper_url.split("/")[-1].split(".pdf")[0]

    # Set the directory to save the downloaded PDF and extracted content
    output_dir = osp.join("data", "pdf")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    pdf_filename = os.path.join(output_dir, f"{paper_id}.pdf")

    command = f"wget -O --user-agent='Mozilla/5.0' {pdf_filename} {paper_url}"

    print(command)

    os.system(command)


import json
import os.path as osp

import pandas as pd
from tqdm import trange

import const
from arguments import parse_args


def download_arXiv_metadata_of_month(existing_arxiv_ids, start_year, end_year, start_month, end_month):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):

            if year == start_year and month < start_month or year == end_year and month > end_month:
                continue

            ids_li = []

            if (month < 4 and year == 2007) or (month <= 11 and year == 2023) or (month > 10 and year >= 2024):
                print(f"SKIP Year {year} Month {month}")
                continue
            else:
                print(f"Adding Year {year} Month {month}")

            prefix = f"{year % 100:02d}{month:02d}"
            # See arXiv monthly submission
            # https://arxiv.org/stats/monthly_submissions

            if year <= 2014:
                ids_li += [f"{prefix}.{x:04d}" for x in range(0, 10000)]

            elif year >= 2015:
                # We just assume that there can be at most 22000 new papers uploaded to arXiv per month
                ids_li += [f"{prefix}.{x:05d}" for x in range(0, 22000)]

            ids_li = sorted(list(set(ids_li) - set(existing_arxiv_ids)))

            additional_data = []

            path_additional_data = osp.join(args.data_dir, "NLP", "arXiv", f"additional_data_{year}_{month}.json")

            INTERVAL = 100

            for idx in trange(0, len(ids_li), INTERVAL, desc=f"Download", position=0, leave=True):
                ids_string = ','.join([f"{x}" for x in ids_li[idx:idx + INTERVAL]])
                url = f'http://export.arxiv.org/api/query?id_list={ids_string}'

                # logger.info(
                #     f"[{category}]\tDownloading {idx}-{idx + 500}/{len(ids_li)}")

                t0 = time.time()

                try:
                    results = feedparser.parse(url)
                    entries = results.entries


                except:
                    print("Cannot get total results")
                    traceback.print_exc()
                    continue

                for result in entries:
                    paper = process_arxiv_entry(result)

                    additional_data += [paper]

                time.sleep(max(0, 3 - (time.time() - t0)))

                if (idx + 1) % 2000 == 0:
                    print(f"Save {len(additional_data)} papers")
                    json.dump(additional_data, open(path_additional_data,
                                                    'w',
                                                    encoding='utf-8'))

            json.dump(additional_data, open(path_additional_data, 'w', encoding='utf-8'))


def add_additional_arXiv_data_to_parquet(df):
    all_additional_data = []

    # Manually specify the additional data to be loaded
    for year in range(2024, 2025):
        for month in range(4, 5):

            if (month < 4 and year == 2007) or (month <= 10 and year == 2023) or (month > 10 and year >= 2024):
                print(f"SKIP Year {year} Month {month}")
                continue
            else:
                print(f"Loading Year {year} Month {month}")

            path_additional_data = osp.join(args.data_dir, "NLP", "arXiv", f"additional_data_{year}_{month}.json")

            if osp.exists(path_additional_data):
                additional_data = json.load(open(path_additional_data, 'r', encoding='utf-8'))
                all_additional_data += additional_data
            else:
                pass

    all_additional_data = pd.DataFrame(all_additional_data)
    all_additional_data = all_additional_data.dropna(subset=[const.ID, const.TITLE, const.SUMMARY])

    df = df.dropna(subset=[const.ID, const.TITLE, const.SUMMARY])

    df = df.reset_index(drop=True)

    df = pd.concat([df, all_additional_data], ignore_index=True)

    for idx in trange(len(df)):
        authors = df.loc[idx, "authors"]
        if isinstance(authors, list):
            df.loc[idx, "authors"] = json.dumps(df.loc[idx, "authors"])
        else:
            assert isinstance(authors, str)

    df.arxiv_comment = df.arxiv_comment.fillna("")
    df = df.astype({
        "arxiv_comment": str,
        const.SUMMARY: str,
        "authors": str,
        "published": str,
        "updated": str,
    })

    df.drop_duplicates(subset=[const.ID], inplace=True)
    df[const.UPDATED] = pd.to_datetime(df[const.UPDATED])
    df = df.sort_values(const.PUBLISHED, ascending=True)

    df.to_parquet(osp.join(args.data_dir, "NLP", "arXiv", "arXiv_metadata.parquet"))


if __name__ == "__main__":
    """
    # Example usage: Download a PDF file of an arXiv paper
    paper_abs_url = "https://arxiv.org/abs/2310.13132"
    paper_id = paper_abs_url.split("/abs/")[-1].split("/")[0]
    print(f"ID: {paper_id}")
    download_arXiv_paper_pdf(paper_id)
    """

    args = parse_args()

    data = load_arXiv_data(args.data_dir)
    existing_arxiv_ids = get_arXiv_IDs_of_existing_papers(data)

    # Example usage: Download arXiv metadata of a specific month
    # download_arXiv_metadata_of_month(existing_arxiv_ids, 2024, 2025,4, 5)

    add_additional_arXiv_data_to_parquet(data)
