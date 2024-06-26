"""
Date: 2024.6.23

Extract keywords from titles OR titles + abstract combined using GPT-3.5-turbo

Run `pip install llama-index` first

"""

import os
import sys
import time
import json

import nest_asyncio
from llama_index.core import Document
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI

sys.path.append(os.path.abspath('.'))

from arguments import parse_args
from utility.utils_data import load_arXiv_data
from utility.utils_misc import project_setup

nest_asyncio.apply()

project_setup()

os.environ["OPENAI_API_KEY"] = "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"


def main():

    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=1024)

    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=512, chunk_overlap=128
    )

    args = parse_args()

    if args.attribute == "title":
        num_keywords = 3
    else:
        num_keywords = 15

    extractors = [
        # TitleExtractor(nodes=5, llm=llm),
        # QuestionsAnsweredExtractor(questions=3, llm=llm),
        # EntityExtractor(prediction_threshold=0.5),
        # SummaryExtractor(summaries=["prev", "self"], llm=llm),
        KeywordExtractor(keywords=num_keywords, llm=llm),
        # CustomExtractor()
    ]

    for start_year in range(args.start_year, args.end_year):

        arxiv_data = load_arXiv_data(args.data_dir, start_year=start_year, start_month=1, end_year=start_year + 1,
                                     end_month=1)

        transformations = [text_splitter] + extractors

        t0 = time.time()

        docs = []

        for i, row in arxiv_data.iterrows():

            if args.attribute == "title":

                docs.append(Document(text=f'{row["title"]}', metadata={"arxiv_id": row["id"]}))

            elif args.attribute == "title_and_abstract":
                docs.append(Document(text=f'{row["title"]}\n{row["summary"]}', metadata={"arxiv_id": row["id"],
                                                                                       "file_name": f"{row['title']}",
                                                                "page_label": 0}))

            else:
                raise ValueError(f"Attribute {args.attribute} not recognized")

        print(f"Creating documents for {args.attribute}: {time.time() - t0:.2f} seconds")


        pipeline = IngestionPipeline(transformations=transformations)

        t0 = time.time()
        nodes = pipeline.run(documents=docs, show_progress=True)

        print(f"Extracting abstracts: {time.time() - t0:.2f} seconds")

        keywords = {entry.metadata['arxiv_id']: entry.metadata['excerpt_keywords'] for entry in nodes}

        print("Saving keywords", end=" ")
        t0 = time.time()

        path = os.path.join(args.data_dir, "NLP", "arXiv", f"{args.attribute}_keywords_{start_year}.json")

        with open(path, "w") as f:
            json.dump(keywords, f)

        print(f"Saving keywords: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()