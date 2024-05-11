"""

# Extracting Metadata for Better Document Indexing and Understanding

In many cases, especially with long documents, a chunk of text may lack the context necessary to disambiguate the chunk from other similar chunks of text. One method of addressing this is manually labelling each chunk in our dataset or knowledge base. However, this can be labour intensive and time consuming for a large number or continually updated set of documents.

To combat this, we use LLMs to extract certain contextual information relevant to the document to better help the retrieval and language models disambiguate similar-looking passages.

We do this through our brand-new `Metadata Extractor` modules.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# from llama_index.legacy.extractors import EntityExtractor
import os
from copy import deepcopy

import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
)
from llama_index.core.schema import MetadataMode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=1024)

"""We create a node parser that extracts the document title and hypothetical question embeddings relevant to the document chunk.

We also show how to instantiate the `SummaryExtractor` and `KeywordExtractor`, as well as how to create your own custom extractor
based on the `BaseExtractor` base class
"""
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)


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
        return metadata_list


extractors = [
    TitleExtractor(nodes=5, llm=llm),
    QuestionsAnsweredExtractor(questions=3, llm=llm),
    # EntityExtractor(prediction_threshold=0.5),
    SummaryExtractor(summaries=["prev", "self"], llm=llm),
    KeywordExtractor(keywords=20, llm=llm),
    # CustomExtractor()
]

transformations = [text_splitter] + extractors

"""We first load the 10k annual SEC report for Uber and Lyft for the years 2019 and 2020 respectively."""

# Note the uninformative document file name, which may be a common scenario in a production setting
uber_docs = SimpleDirectoryReader(input_files=["data/uber.pdf"]).load_data()
uber_front_pages = uber_docs[0:3]
uber_content = uber_docs[63:69]
uber_docs = uber_front_pages + uber_content

pipeline = IngestionPipeline(transformations=transformations)

uber_nodes = pipeline.run(documents=uber_docs)

uber_nodes[1].metadata

# Note the uninformative document file name, which may be a common scenario in a production setting
lyft_docs = SimpleDirectoryReader(
    input_files=["data/lyft.pdf"]
).load_data()
lyft_front_pages = lyft_docs[0:3]
lyft_content = lyft_docs[68:73]
lyft_docs = lyft_front_pages + lyft_content

pipeline = IngestionPipeline(transformations=transformations)

lyft_nodes = pipeline.run(documents=lyft_docs)

lyft_nodes[2].metadata

"""Since we are asking fairly sophisticated questions, we utilize a subquestion query engine for all QnA pipelines below, and prompt it to pay more attention to the relevance of the retrieved sources."""

question_gen = LLMQuestionGenerator.from_defaults(
    llm=llm,
    prompt_template_str="""
        Follow the example, but instead of giving a question, always prefix the question
        with: 'By first identifying and quoting the most relevant sources, '.
        """
                        + DEFAULT_SUB_QUESTION_PROMPT_TMPL,
)

"""## Querying an Index With No Extra Metadata"""

nodes_no_metadata = deepcopy(uber_nodes) + deepcopy(lyft_nodes)
for node in nodes_no_metadata:
    node.metadata = {
        k: node.metadata[k]
        for k in node.metadata
        if k in ["page_label", "file_name"]
    }
print(
    "LLM sees:\n",
    (nodes_no_metadata)[9].get_content(metadata_mode=MetadataMode.LLM),
)

index_no_metadata = VectorStoreIndex(
    nodes=nodes_no_metadata,
)
engine_no_metadata = index_no_metadata.as_query_engine(
    similarity_top_k=10, llm=OpenAI(model="gpt-4")
)

final_engine_no_metadata = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=engine_no_metadata,
            metadata=ToolMetadata(
                name="sec_filing_documents",
                description="financial information on companies",
            ),
        )
    ],
    question_gen=question_gen,
    use_async=True,
)

response_no_metadata = final_engine_no_metadata.query(
    """
    What was the cost due to research and development v.s. sales and marketing for uber and lyft in 2019 in millions of USD?
    Give your answer as a JSON.
    """
)
print(response_no_metadata.response)
# Correct answer:
# {"Uber": {"Research and Development": 4836, "Sales and Marketing": 4626},
#  "Lyft": {"Research and Development": 1505.6, "Sales and Marketing": 814 }}

"""**RESULT**: As we can see, the QnA agent does not seem to know where to look for the right documents. As a result it gets the Lyft and Uber data completely mixed up.

## Querying an Index With Extracted Metadata
"""

print(
    "LLM sees:\n",
    (uber_nodes + lyft_nodes)[9].get_content(metadata_mode=MetadataMode.LLM),
)

index = VectorStoreIndex(
    nodes=uber_nodes + lyft_nodes,
)
engine = index.as_query_engine(similarity_top_k=10, llm=OpenAI(model="gpt-4"))

final_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name="sec_filing_documents",
                description="financial information on companies.",
            ),
        )
    ],
    question_gen=question_gen,
    use_async=True,
)

response = final_engine.query(
    """
    What was the cost due to research and development v.s. sales and marketing for uber and lyft in 2019 in millions of USD?
    Give your answer as a JSON.
    """
)
print(response.response)
# Correct answer:
# {"Uber": {"Research and Development": 4836, "Sales and Marketing": 4626},
#  "Lyft": {"Research and Development": 1505.6, "Sales and Marketing": 814 }}

"""**RESULT**: As we can see, the LLM answers the questions correctly.

### Challenges Identified in the Problem Domain

In this example, we observed that the search quality as provided by vector embeddings was rather poor. This was likely due to highly dense financial documents that were likely not representative of the training set for the model.

In order to improve the search quality, other methods of neural search that employ more keyword-based approaches may help, such as ColBERTv2/PLAID. In particular, this would help in matching on particular keywords to identify high-relevance chunks.

Other valid steps may include utilizing models that are fine-tuned on financial datasets such as Bloomberg GPT.

Finally, we can help to further enrich the metadata by providing more contextual information regarding the surrounding context that the chunk is located in.

### Improvements to this Example
Generally, this example can be improved further with more rigorous evaluation of both the metadata extraction accuracy, and the accuracy and recall of the QnA pipeline. Further, incorporating a larger set of documents as well as the full length documents, which may provide more confounding passages that are difficult to disambiguate, could further stresss test the system we have built and suggest further improvements.
"""
