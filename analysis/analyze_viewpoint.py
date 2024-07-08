import json
import pickle

from tqdm import tqdm
import os.path as osp

from arguments import parse_args
from utility.utils_data import load_keywords
from utility.utils_misc import project_setup

related_keywords = {
    "lgbt": [
        "lesbian",
        "gay",
        "bisexual",
        "transgender",
        "queer",
        "sexual orientation",
        "gender identity",
        "lgbt rights",
        "lgbt health",
        "same-sex marriage",
        "lgbt youth",
        "gender dysphoria",
        "lgbt communities",
        "lgbt discrimination",
        "lgbt mental health",
        "queer theory",
        "coming out",
        "sexual minorities",
        "homophobia",
        "transphobia"
    ],
    "minority": [
        "human rights",
        "racial minority",
        "ethnic minority",
        "cultural diversity",
        "minority health",
        "minority rights",
        "socioeconomic status",
        "minority education",
        "minority representation",
        "discrimination",
        "marginalization",
        "minority identity",
        "intersectionality",
        "minority groups",
        "immigrant communities",
        "social inequality",
        "minority empowerment",
        "health disparities",
        "minority politics",
        "underrepresented groups",
        "minority integration"
    ],
    "dei": [
        "diversity",
        "equity",
        "inclusion",
        "inclusive practices",
        "social justice",
        "equal opportunity",
        "equity in education",
        "workplace diversity",
        "inclusive education",
        "diversity management",
        "equity policies",
        "inclusive leadership",
        "cultural competence",
        "bias reduction",
        "gender equity",
        "racial equity",
        "inclusive policies",
        "belonging",
        "dei strategies",
        "diversity training",
        "inclusive organizations",
        "equity initiatives",
        "social inclusion",
        "disparities",
        "institutional racism",
        "affirmative action",
        "inclusive research",
        "equity in healthcare",
        "diversity metrics",
        "accessibility"
    ],
    "GAI": [
        "artificial general intelligence",
        "general ai",
        "autonomous intelligence",
        "ai singularity",
        "ai consciousness",
        "general-purpose ai",
        "universal ai",
        "machine consciousness",
        "general intelligence",
        "ai self-awareness",
        "ai safety",
        "ai risk",
        "ai threats",
        "ai ethics",
        "superintelligent ai",
        "ai governance",
        "ai alignment",
        "ai regulation",
        "existential risk",
        "human-ai coexistence",
        "ai impact",
        "ai oversight",
        "general ai challenges",
        "ai autonomy",
        "ai policy"
    ],

    "Open Access": [
        "open access",
        "academic publishing",
        "peer review",
        "scholarly communication",
        "article processing charges",
        "apc"
        "institutional repositories",
        "preprints",
        "gold open access",
        "green open access",
        "hybrid journals",
        "copyright",
        "licensing",
        "creative commons (cc)",
        "self-archiving",
        "public access policies",
        "research data sharing",
        "impact factor",
        "citation metrics",
        "digital preservation",
        "subscription model",
        "journal impact",
        "scholarly journals",
        "editorial policies",
        "publication ethics",
        "author rights",
        "digital libraries",
        "metadata",
        "indexing services",
        "academic collaboration",
        "knowledge dissemination"
    ]

}

if __name__ == "__main__":
    project_setup()
    args = parse_args()

    keywords = load_keywords(args.data_dir, args.feature_name)
    keywords['year'] = keywords['published'].dt.year

    # Group by keyword and years, then aggregate the paper IDs
    keyword_year_papers = keywords.reset_index().explode('keywords').groupby(['keywords', 'year'])['id'].apply(
        list).reset_index()

    # Convert to a nested dictionary
    keyword_to_year_papers = {}
    for _, row in tqdm(keyword_year_papers.iterrows()):
        keyword = row['keywords']
        year = row['year']
        papers = row['id']

        if keyword not in keyword_to_year_papers:
            keyword_to_year_papers[keyword] = {}
        keyword_to_year_papers[keyword][year] = papers


    with open(osp.join(args.output_dir, "keyword_to_year_papers.parquet"), "wb") as f:
        pickle.dump(keyword_to_year_papers, f)



    # Print the results
    print(keyword_to_papers)
