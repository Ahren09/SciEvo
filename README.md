# arXiv Data


This repository contains all the abstracts of arXiv. 

## Dataset Description



arXiv has a comprehensive and intricate taxonomy that categorizes research papers into broadly 8 fields, including Computer Science (cs), Economics (econ), Electrical Engineering and Systems Science (eess), Mathematics (math), Physics, Quantitative Biology (q-bio), Quantitative Finance (q-fin), and  Statistics (stats).

Within Computer Science, there are several subfields, each dedicated to specific areas of research and innovation. For example, Artificial Intelligence (cs.AI),Machine Learning (cs.ML), Computer Vision (cs.CV), etc.

More information can be found in **[arXiv Submission Taxonomy](https://arxiv.org/category_taxonomy)**.


- `arXiv_metadata.pkl`: The full dataaset with around 2 million entries. Note that the `tags` field in this dataframe contains tags of MSC Classes (e.g. `I.4.5`, `52B12`, `05A15`)
  - To load this file, use `utility.utils_data.load_data(args)`
  - Alternatively, you can directly use `pd.read_pickle('arXiv_metadata.pkl')`.

- `arXiv_metadata_last_100_entries.xlsx`: contains the last 100 entries in `arXiv_metadata.pkl`
- `arXiv_metadata_last_10000_entries.xlsx`: contains the last 10000 entries in `arXiv_metadata.pkl`


## Getting Started

Download the data file `arXiv_metadata.pkl` and save it into the `data/` directory.


## Download arXiv Papers

To download the archive papers you can use `dataset/download_arxiv_paper.py`. Alternatively, archive provides bulk download using `s3cmd`. Refer to the [arXiv Bulk Data Access](https://info.arxiv.org/help/bulk_data.html) for more information.




