from datasets import Dataset, load_dataset
import json
import os

# 1. Your original dictionary
raw_dict = json.load(open(os.path.expanduser("~/data/NLP/arXiv/title_keywords.json"), "r"))

# 2. Convert to list of dicts with 'id' and 'keywords'
data = [
    {
        "id": key,
        "keywords": [kw.strip() for kw in value.split(",")]
    }
    for key, value in raw_dict.items()
]

# 3. Create a Hugging Face Dataset
new_subset = Dataset.from_list(data)

# Optional: preview
print(new_subset[0])
# {'id': 'http://arxiv.org/abs/physics/9403001', 'keywords': ['Superstrings', 'String theory', 'Quantum gravity']}


# 5. Push as a new subset (e.g., 'title_keywords') to your dataset repo
new_subset.push_to_hub("anonymous/SciEvo", split="title_keywords")
