import json
import os
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor

import const
from utility.utils_data import load_data

sys.path.insert(0, os.path.abspath('..'))
from arguments import parse_args
from utility.utils_misc import project_setup


def main():
    df = load_data(args)

    def populate_tag2papers(proc_id, start, end, local_dict):
        for i, row in df.iloc[start:end].iterrows():
            if (i + 1 - start) % 10000 == 0:
                print(f"[Process {proc_id}] Processed {i + 1 - start}/{end - start} rows")

            tags = [tag for tag in row['tags'] if tag in
                    const.ARXIV_SUBJECTS_LIST]
            for tag in tags:

                if tag not in local_dict:
                    local_dict[tag] = []
                local_dict[tag].append(row[const.ID])

    # Create empty dictionary to store tags and corresponding paper indices
    tag2papers = {}

    # Create multiple smaller dictionaries to store intermediate results
    dicts = [{} for _ in range(args.num_workers)]

    # Number of rows to process in each thread
    step = len(df) // args.num_workers

    # List to store future results
    futures = []

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        for i in range(args.num_workers):
            start_index = i * step
            end_index = (i + 1) * step if i != (args.num_workers - 1) else len(df)
            futures.append(executor.submit(populate_tag2papers, i, start_index, end_index, dicts[i]))

    # Wait for all threads to complete and merge the dictionaries
    for future in futures:
        future.result()

    for d in dicts:
        for key, value in d.items():
            if key not in tag2papers:
                tag2papers[key] = []
            tag2papers[key].extend(value)

    for key in tag2papers:
        tag2papers[key] = sorted(list(set(tag2papers[key])))

    print(f"Saving tag2papers with {len(tag2papers)} unique tags ...", end=" ")
    json.dump(tag2papers, open(osp.join(args.data_dir, "tag2papers.json"), 'w', encoding='utf-8'), indent=2)
    print("done!")



if __name__ == "__main__":
    project_setup()
    args = parse_args()
    main()
