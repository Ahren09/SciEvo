"""Constants and configuration values for SciEvo package.

This module contains all constant values used throughout the SciEvo package,
including arXiv category definitions, field mappings, and academic stop words.
"""

from typing import Dict, List, Set

# Dataset field names
ID = "id"
PAPERID = "paperId"
TITLE = "title_llm_extracted_keyword"
SUMMARY = "summary"
UPDATED = "updated"
PUBLISHED = "published"
PUBLICATIONDATE = "publicationDate"
SUBJECT = "subject"
FOS = "fieldsOfStudy"

# Graph edge definitions
SOURCE = "source"
DESTINATION = "destination"

# Model names
WORD2VEC = "word2vec"
GCN = "gcn"

# Date format string
DATE_FORMAT = "%Y-%m-%d"

# arXiv category definitions
ARXIV_CATEGORIES_CS: List[str] = [
    'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL',
    'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL',
    'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR',
    'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
    'cs.MA', 'cs.ML', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE',
    'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO',
    'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY'
]

ARXIV_CATEGORIES_ECON: List[str] = ['econ.EM', 'econ.GN', 'econ.TH']

ARXIV_CATEGORIES_EESS: List[str] = ['eess.AS', 'eess.IV', 'eess.SP', 'eess.SY']

ARXIV_CATEGORIES_MATH: List[str] = [
    'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA',
    'math.CO', 'math.CT', 'math.CV', 'math.DG', 'math.DS',
    'math.FA', 'math.GM', 'math.GN', 'math.GR', 'math.GT',
    'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MG',
    'math.MP', 'math.NA', 'math.NT', 'math.OA', 'math.OC',
    'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG',
    'math.SP', 'math.ST'
]

ARXIV_CATEGORIES_PHYSICS: List[str] = [
    'astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA',
    'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR',
    'cond-mat.dis-nn', 'cond-mat.mes-hall',
    'cond-mat.mtrl-sci', 'cond-mat.other',
    'cond-mat.quant-gas', 'cond-mat.soft',
    'cond-mat.stat-mech', 'cond-mat.str-el',
    'cond-mat.supr-con', 'gr-qc', 'hep-ex', 'hep-lat',
    'hep-ph', 'hep-th', 'math-ph', 'nlin.AO', 'nlin.CD',
    'nlin.CG', 'nlin.PS', 'nlin.SI', 'nucl-ex',
    'nucl-th', 'physics.acc-ph', 'physics.ao-ph',
    'physics.app-ph', 'physics.atm-clus',
    'physics.atom-ph', 'physics.bio-ph',
    'physics.chem-ph', 'physics.class-ph',
    'physics.comp-ph', 'physics.data-an',
    'physics.ed-ph', 'physics.flu-dyn',
    'physics.gen-ph', 'physics.geo-ph',
    'physics.hist-ph', 'physics.ins-det',
    'physics.med-ph', 'physics.optics',
    'physics.plasm-ph', 'physics.pop-ph',
    'physics.soc-ph', 'physics.space-ph', 'quant-ph'
]

ARXIV_CATEGORIES_QBIO: List[str] = [
    'q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN',
    'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM',
    'q-bio.SC', 'q-bio.TO'
]

ARXIV_CATEGORIES_QFIN: List[str] = [
    'q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF',
    'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST',
    'q-fin.TR'
]

ARXIV_CATEGORIES_STAT: List[str] = [
    'stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT',
    'stat.TH'
]

# Subject to categories mapping
ARXIV_SUBJECTS: Dict[str, List[str]] = {
    "econ": ARXIV_CATEGORIES_ECON,
    "q-bio": ARXIV_CATEGORIES_QBIO,
    "q-fin": ARXIV_CATEGORIES_QFIN,
    "stat": ARXIV_CATEGORIES_STAT,
    "eess": ARXIV_CATEGORIES_EESS,
    "math": ARXIV_CATEGORIES_MATH,
    "cs": ARXIV_CATEGORIES_CS,
    "physics": ARXIV_CATEGORIES_PHYSICS,
}

# Category to subject reverse mapping
ARXIV_CATEGORIES_TO_SUBJECT: Dict[str, str] = {
    cat: subject for subject, cats in ARXIV_SUBJECTS.items() for cat in cats
}

# All arXiv subjects as a flat list
ARXIV_SUBJECTS_LIST: List[str] = [
    tag for name, cat in ARXIV_SUBJECTS.items() for tag in cat
]

# Academic stop words for text processing
ACADEMIC_STOP_WORDS: List[str] = [
    '1', '2', 'also', 'approach', 'data',
    'existing', 'feature', 'final', 'finally', 'find',
    'given', 'method', 'methods', 'model', 'models',
    'new', 'novel', 'paper', 'performance', 'present', 'problem',
    'propose', 'proposed', 'prove', 'proved', 'provide', 'provides', 
    'recent', 'recently', 'research', 'results', 's', 'several',
    'show', 'shows', 'shown', 'showed', 'state-of-the-art', 'study',
    'task', 'tasks', 'use', 'used', 'using', 'various', 'via', 
    'what', 'whose', 'when', 'with', 'without'
]

# Add numeric and alphabetic characters to stop words
ACADEMIC_STOP_WORDS += [str(i) for i in range(10)]
all_chars = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
ACADEMIC_STOP_WORDS += [c for c in all_chars]

# Semantic Scholar statistics
SEMANTIC_SCHOLAR_STATS: Dict[str, float] = {
    "n/a": 60.0e6,
    "Medicine": 31.8e6,
    "Biology": 20.4e6,
    "Physics": 11.6e6,
    "Engineering": 10.2e6,
    "Computer Science": 9.7e6,
    "Chemistry": 9.1e6,
    "Education": 7.4e6,
    "Materials Science": 7.4e6,
    "Environmental Science": 7.0e6,
    "Economics": 6.2e6,
    "Psychology": 6.2e6,
    "Agricultural and Food Sciences": 5.9e6,
    "Business": 5.6e6,
    "Mathematics": 3.7e6,
    "History": 3.4e6,
    "Political Science": 2.9e6,
    "Art": 2.8e6,
    "Geology": 2.6e6,
    "Sociology": 1.4e6,
    "Philosophy": 1.4e6,
    "Law": 1.1e6,
    "Linguistics": 1.1e6,
    "Geography": 0.35e6
}

# arXiv category keywords for specific domains
ARXIV_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    'cs.ai': [
        'adversarial', 'algorithm', 'bayesian', 'classification', 'clustering',
        'decision tree', 'deep learning', 'dimensionality', 'evolutionary',
        'feature selection', 'gan', 'machine learning', 'model', 'naive bayes', 
        'neural', 'optimization', 'reduction', 'regression', 'reinforcement learning', 
        'rl', 'supervised', 'support vector machine', 'svm', 'unsupervised'
    ],
    'cs.cl': [
        'bert', 'chatgpt', 'classification', 'dialogue', 'extraction', 'generation', 
        'gpt', 'information', 'language', 'model', 'lemmatization', 'llm', 
        'named entity recognition', 'ner', 'nlp', 'parsing', 'pos tagging', 
        'regression', 'semantic', 'sentiment analysis', 'speech', 'stemming',
        'summarization', 'syntactic parsing', 'text', 'text mining', 'tokenization', 
        'transformer', 'translation'
    ],
    'cs.cv': [
        'classification', 'cnn', 'convolutional neural network', 'cv', 'detection', 
        'face', 'feature', 'image', 'imagenet', 'object recognition', 'optical flow', 
        'rcnn', 'recognition', 'regression', 'resnet', 'resolution', 'representation', 
        'scene', 'segmentation', 'ssd', 'texture', 'video', 'vision', 'yolo'
    ],
    'cs.lg': [
        'active learning', 'backpropagation', 'bagging', 'boosting', 'cross validation', 
        'deep learning', 'descent', 'dropout', 'ensemble', 'feature', 'gradient',
        'learning', 'loss', 'machine learning', 'model', 'neural network', 'offline', 
        'online', 'optimization', 'overfitting', 'regularization', 'reinforcement learning', 
        'representation', 'statistical learning', 'supervised', 'transfer learning', 
        'unsupervised'
    ],
    'cs.ir': [
        'behavior', 'click', 'clickthrough', 'cf', 'collaborative', 'content', 
        'document', 'factorization', 'filtering', 'frequency', 'hybrid', 'image', 
        'indexing', 'item', 'information', 'mf', 'model', 'pagerank', 'query',
        'ranking', 'recommendation', 'recommender', 'relevance', 'representation', 
        'retrieval', 'search', 'text', 'term', 'user', 'vector', 'video', 'web'
    ]
}

# Subject to keywords mapping for research topic identification
SUBJECT2KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    'Computer Science': {
        'large language models': [
            'large language models', 'large language model', 'llm', 'llms', 'gpt', 
            'chatgpt', 'gpt4', 'gpt3', 'gpt-4', 'gpt-3', 'rlhf', 'chain-of-thought', 
            'chain of thought', 'chain of thoughts', 'cot', 'instruction-tuning', 
            'instruction tuning', 'retrieval augmented generation', 
            'retrieval-augmented generation', 'rag'
        ],
        'machine learning': [
            'machine learning', 'artificial intelligence', 'deep learning', 'neural networks'
        ],
        'edge computing': [
            'edge computing', 'distributed systems', 'cloud computing', 'iot'
        ],
        'natural language processing': [
            'natural language processing', 'machine translation', 'text', 'speech'
        ],
        'quantum computing': [
            'quantum algorithms', 'quantum', 'quantum computing', 'computational complexity',
            'quantum cryptography'
        ],
        'language models': [
            'bert', 'roberta', 'xlnet', 'transformer', 'transformers', 'bidirectional', 
            'attention', 'self-attention', 'language models'
        ],
        'reinforcement learning': [
            'reinforcement learning', 'reinforcement', 'dpo', 'policy', 'ppo', 
            'decision making', 'reinforce', 'rl', 'rlhf', 'mdp', 'markov'
        ],
        'HMM': [
            'hmm', 'markov', 'markov chain', 'markov model', 'markov decision', 'markov process'
        ],
        'computer vision': [
            'computer vision', 'multimodal', 'multimodality', 'visual', 'vision', 
            'visualness', 'perception', 'yolo', 'cnn', 'cnns', 'convolutional', 
            'convolution', 'object detection', 'object recognition'
        ],
        'graph neural networks': [
            'graph neural networks', 'gnn', 'gcn', 'gat', 'graphsage', 'graph mining', 
            'gnns', 'network analysis', 'geometric'
        ],
        'social computing': [
            'social computing', 'computational social science', 'social networks', 
            'social media', 'twitter', 'reddit', 'social'
        ],
        'Cybersecurity': ['cybersecurity', 'security', 'privacy', 'cryptography'],
        'Bias': [
            'bias', 'fairness', 'ethics', 'ethical', 'fair', 'bias', 'equality', 
            'fairness', 'equity'
        ],
        'Adversarial': ['adversarial', 'robustness', 'defense', 'adversarial training'],
        'Explainable AI': [
            'explainable', 'interpretability', 'interpretation', 'interpretable', 
            'explanation', 'explainability', 'xai'
        ],
        'Multi-lingual Applications': [
            'multilingual', 'cross-lingual', 'translation', 'mt'
        ],
        'named entity': [
            'named entity', 'ner', 'entity recognition', 'srl', 'tagging', 'pos', 
            'part-of-speech'
        ],
        'question answering': [
            'question answering', 'qa', 'question', 'answering', 'q&a'
        ],
        'CNN': ['cnn', 'cnns', 'convolutional', 'convolution'],
        'recommender systems': [
            'recommendation', 'recommender', 'matrix factorization', 'mf', 'cf',
            'collaborative filtering', 'ncf', 'ngcf', 'lightgcn', 'bpr', 'two-tower'
        ]
    },
    'Education': {
        'hybrid teaching': [
            'hybrid teaching', 'blended learning', 'online', 'educational technology', 
            'pedagogical', 'pedagogy', 'e-learning'
        ],
        'cognitive strategies': [
            'cognitive strategies', 'metacognition', 'educational psychology', 
            'learning theories'
        ],
        'quantitative research': [
            'quantitative research', 'educational statistics', 'data analysis', 'statistics'
        ],
        'data mining': [
            'data mining', 'big data', 'learning analytics', 'adaptive learning', 
            'statistics', 'data analysis'
        ],
        'Multiculture': [
            'multicultural', 'multiculture', 'diversity', 'inclusive', 'inclusion', 
            'cross-cultural'
        ]
    },
    'Business': {
        'consumer neuroscience': [
            'consumer neuroscience', 'neuromarketing', 'decision making', 'consumer behavior'
        ],
        'Sustainability': [
            'sustainable', 'sustainability', 'green logistics', 'social responsibility', 
            'environmental management', 'environmental protection'
        ],
        'Fintech': ['fintech', 'blockchain', 'digital payments', 'regtech'],
        'behavioral economics': [
            'behavioral economics', 'consumer psychology', 'market research', 'behavior', 
            'behavioral', 'economic decision-making'
        ],
        'corporate governance': [
            'corporate governance', 'emerging markets', 'international', 'global market', 
            'market regulation'
        ]
    },
    'Medicine': {
        'precision medicine': [
            'precision medicine', 'precision', 'genomic medicine', 'personalized', 
            'pharmacogenomics', 'biomarker discovery', 'targeted', 'individualized', 
            'predictive'
        ],
        'Immunotherapy': [
            'immunotherapy', 'cancer', 'checkpoint', 'car', 't-cell', 'immune', 
            'modulators', 'monoclonal antibodies', 'vaccine therapy', 'blockade', 
            'adoptive cell transfer', 'cytokine', 't-cell therapy', 'lymphoma treatment', 
            'biotechnology'
        ],
        'digital health': [
            'digital health', 'telehealth', 'telemedicine', 'wearable devices', 
            'healthcare analytics'
        ],
        'COVID-19': [
            'covid', 'covid-19', 'sars-cov-2', 'coronavirus', 'pandemic', 
            'post-acute sequelae'
        ],
        'microbiome therapeutics': [
            'microbiome therapeutics', 'gut-brain axis', 'probiotics', 'microbial ecology'
        ],
        'neurodegenerative': [
            'neurodegenerative', 'alzheimer', 'parkinson', 'tauopathy', 'tauopathies'
        ],
        'stem cell therapy': [
            'stem cell therapy', 'regenerative medicine', 'tissue engineering', 
            'cell therapy', 'bone marrow transplant', 'pluripotent stem cells'
        ],
        'Radiology': [
            'radiology', 'imaging', 'MRI', 'CT scan', 'ultrasound', 'nuclear medicine', 
            'diagnostic imaging'
        ],
        'Epidemiology': [
            'epidemiology', 'public health', 'disease surveillance', 'vaccination programs', 
            'health policy', 'outbreak response'
        ],
        'aging': [
            'aging', 'gerontology', 'longevity', 'senescence', 'anti-aging therapies', 
            'age-related diseases'
        ],
        'mental health': [
            'mental health', 'psychiatry', 'depression', 'anxiety', 'PTSD', 
            'cognitive behavioral therapy', 'psychotherapy'
        ]
    },
    'Biology': {
        'genome engineering': [
            'genome engineering', 'crispr', 'crispr-cas9', 'gene editing', 'gene therapy',
            'molecular cloning'
        ],
        'synthetic biology': [
            'synthetic biology', 'synthetic genomes', 'artificial life', 'genetic synthesis',
            'bioengineering'
        ],
        'Biodiversity': [
            'environmental dna', 'biodiversity', 'ecological surveying', 'conservation genetics'
        ],
        'bioinformatics': [
            'bioinformatics', 'phylogenetics', 'genomic sequencing', 'taxonomy'
        ],
        'Neuroscience': [
            'neuroscience', 'neural circuits', 'neurobiology', 'brain function', 'cognitive'
        ],
        'Plant Biology': [
            'plant biology', 'photosynthesis', 'botany', 'plant genetics', 
            'agricultural science', 'crop engineering'
        ],
        'marine biology': [
            'marine biology', 'oceanography', 'aquatic ecosystems', 'coral reefs', 
            'marine conservation', 'deep-sea research'
        ],
        'developmental biology': [
            'developmental biology', 'embryology', 'organogenesis', 'morphogenesis', 
            'growth factors', 'stem cell niches'
        ],
        'evolutionary biology': [
            'evolutionary biology', 'natural selection', 'speciation', 'phylogeny',
            'adaptive evolution', 'population genetics'
        ],
        'Microbiology': [
            'microbiology', 'bacteriology', 'virology', 'pathogens', 'antibiotics', 
            'infectious diseases', 'microbial resistance'
        ]
    },
    'Physics': {
        'quantum mechanics': [
            'quantum', 'quantum mechanics', 'quantum field', 'quantum algorithms'
        ],
        'dark energy': [
            'dark energy', 'dark matter', 'cosmology', 'gravitational waves', 'black holes'
        ],
        'particle physics': ['particle physics', 'neutrino', 'standard model'],
        'graphene': [
            'graphene', 'material science', 'nanotechnology', 'semiconductors', 
            'superconductors'
        ],
        'topological insulators': [
            'topological insulators', 'quantum physics', 'material properties', 
            'electronic band'
        ],
        'Cosmology': ['ligo', 'gravitational waves', 'astrophysics', 'cosmology']
    },
    'Engineering': {
        'solar energy': [
            'perovskite', 'solar cells', 'photovoltaics', 'solar energy', 'nanomaterials'
        ],
        'internet of things': [
            'internet of things', 'iotcybersecurity', 'network security', 'smart devices'
        ],
        '5G': ['5g', 'wireless', 'mobile', 'network infrastructure', 'edge computing'],
        'Biomimetic': [
            'biomimetic', 'biomimetics', 'bio-inspired design', 'material science', 
            'sustainable technology'
        ],
        'additive manufacturing': [
            'additive manufacturing', '3d printing', 'prototyping', 'prototype',
            'industrial manufacturing'
        ]
    },
    'Mathematics': {
        'algebraic geometry': [
            'algebraic geometry', 'complex manifolds', 'commutative algebra', 'number theory'
        ],
        'Optimization': [
            'optimization', 'optimize', 'statistical learning', 'algorithmic efficiency',
            'computational efficiency'
        ],
        'PDE': [
            'pde', 'differential equations', 'mathematical modeling', 'chaos theory', 
            'fluid dynamics'
        ],
        'Graph Theory': [
            'random graphs', 'graph', 'graphs', 'networks', 'probability theory', 
            'network science', 'statistical mechanics'
        ],
        'probability': [
            'probability', 'stochastic', 'random processes', 'markov', 'mcmc', 'martingales'
        ],
        'Mathematical Biology': [
            'mathematical biology', 'biostatistics', 'epidemiology', 'population'
        ]
    },
    'History': {
        'Narrative History': [
            'microhistory', 'narrative history', 'cultural history', 'social structures',
            'historiography'
        ],
        'transnational': [
            'transnational', 'migration', 'globalization', 'cultural exchange', 'diaspora'
        ],
        'Pandemics': [
            'pandemics', 'epidemics', 'epidemiology', 'medical history', 'public health', 
            'disease'
        ],
        'Digital Humanities': [
            'archive', 'archival', 'digital', 'historiography', 'digital humanities', 
            'archival science'
        ],
        'Oral History': [
            'oral history', 'memory', 'narrative analysis', 'cultural memory', 
            'historical methodology'
        ]
    }
}

# Keyword to ID mapping for topic identification
KEYWORD2ID: Dict[str, int] = {}
for subject in SUBJECT2KEYWORDS:
    for topic, keywords in SUBJECT2KEYWORDS[subject].items():
        # Ensure no duplicate topics
        assert topic not in KEYWORD2ID, f"Duplicate topic: {topic}"
        KEYWORD2ID[topic] = len(KEYWORD2ID)