# arXiv Data


This repository contains the bibliographic data for all the arXiv papers released until April 21st. 

## Dataset Description



arXiv has a comprehensive taxonomy that categorizes research papers into broadly 8 fields, including Computer Science (cs), Economics (econ), Electrical Engineering and Systems Science (eess), Mathematics (math), Physics, Quantitative Biology (q-bio), Quantitative Finance (q-fin), and  Statistics (stats).

Within Computer Science, there are several subfields, each dedicated to specific areas of research and innovation. For example, Artificial Intelligence (cs.AI),Machine Learning (cs.ML), Computer Vision (cs.CV), etc.

More information can be found in **[arXiv Submission Taxonomy](https://arxiv.org/category_taxonomy)**.


- `arXiv_metadata.parquet`: The full dataaset with >2 million entries. Note that the `tags` field in this dataframe contains tags of MSC Classes (e.g. `I.4.5`, `52B12`, `05A15`). We can ignore these tags for now
  - To load this file, use `utility.utils_data.load_data(DATA_DIR)`
- `arXiv_metadata_last_100_entries.xlsx`: contains the last 100 entries in `arXiv_metadata.parquet`


## Getting Started

1. Download the data file `arXiv_metadata.parquet` and save it into the `data/` directory.
   
   Download the [citation data](https://www.dropbox.com/scl/fo/2005u201x6gmdp44vqdkf/AFSIjfUPB87Ew9_mgwjT8Gc?rlkey=jykfw2rm3p38bwex44rs4hcm7&dl=0) from Dropbox:

   NOTE: The link will expire on July 22, 2024. A new link will be provided by that time.

   

2. Create a new environment

  ```bash
  conda create -n arxiv
  conda activate arxiv
  ```

  Install the dependencies using pip
  
  ```bash
  pip install -r requirements.txt
  ```


## Download arXiv Papers

To download the PDFs of arXiv papers, you can use the functions in `dataset/download_arxiv_paper.py`. 

Alternatively, arXiv provides bulk download using `s3cmd`. Refer to the [arXiv Bulk Data Access](https://info.arxiv.org/help/bulk_data.html) for more information.




