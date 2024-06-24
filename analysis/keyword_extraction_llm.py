import os
import time

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

from arguments import parse_args
from utility.utils_data import load_arXiv_data
from utility.utils_misc import project_setup

nest_asyncio.apply()

project_setup()

os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=1024)

text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

args = parse_args()

class CustomExtractor(BaseExtractor):
    def extract(self, nodes):
        metadata_list = [
            {
                "custom": (
                        node.metadata["document_title"]
                        + "\n"
                        + node.metadata["excerpt_keywords"]
                )
            }
            for node in nodes
        ]
        return metadata_lists


extractors = [
    # TitleExtractor(nodes=5, llm=llm),
    # QuestionsAnsweredExtractor(questions=3, llm=llm),
    # EntityExtractor(prediction_threshold=0.5),
    # SummaryExtractor(summaries=["prev", "self"], llm=llm),
    KeywordExtractor(keywords=20, llm=llm),
    # CustomExtractor()
]

arxiv_data = load_arXiv_data(args.data_dir)

transformations = [text_splitter] + extractors

t0 = time.time()

titles, abstracts = [], []

for i, row in arxiv_data.iterrows():
    title = row["title"]
    abstract = row["summary"]
    titles.append(Document(text=title, metadata={"arxiv_id": row["id"], "file_name": f"{row['title']}", "page_label": 0}))
    abstracts.append(Document(text=abstract, metadata={"arxiv_id": row["id"], "file_name": f"{row['title']}",
                                                       "page_label": 0}))

    if i == 100:
        break

# titles = arxiv_data.iloc[:1000]["title"].apply(lambda x: Document(text=x))
# abstracts = arxiv_data.iloc[:1000]["summary"].apply(lambda x: Document(text=x))

print(f"Creating documents: {time.time() - t0:.2f} seconds")


pipeline = IngestionPipeline(transformations=transformations)

t0 = time.time()
abstracts_nodes = pipeline.run(documents=abstracts, num_workers=32)

print(f"Creating documents: {time.time() - t0:.2f} seconds")
print(abstracts_nodes[0].metadata)


