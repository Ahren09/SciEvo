ID = "id"
PAPERID = "paperId"
TITLE = "title"
SUMMARY = "summary"
UPDATED = "updated"
PUBLISHED = "published"
PUBLICATIONDATE = "publicationDate"
SUBJECT = "subject"
FOS = "fieldsOfStudy"

# Model Names
WORD2VEC = "word2vec"

format_string = "%Y-%m-%d"


ARXIV_CATEGORIES_CS = ['cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL',
                       'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL',
                       'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR',
                       'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
                       'cs.MA', 'cs.ML', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE',
                       'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO',
                       'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY']

ARXIV_CATEGORIES_ECON = ['econ.EM', 'econ.GN', 'econ.TH']

ARXIV_CATEGORIES_EESS = ['eess.AS', 'eess.IV', 'eess.SP', 'eess.SY']

ARXIV_CATEGORIES_MATH = ['math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA',
                         'math.CO', 'math.CT', 'math.CV', 'math.DG', 'math.DS',
                         'math.FA', 'math.GM', 'math.GN', 'math.GR', 'math.GT',
                         'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MG',
                         'math.MP', 'math.NA', 'math.NT', 'math.OA', 'math.OC',
                         'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG',
                         'math.SP', 'math.ST']

ARXIV_CATEGORIES_PHYSICS = ['astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA',
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
                            'physics.soc-ph', 'physics.space-ph', 'quant-ph']

ARXIV_CATEGORIES_QBIO = ['q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN',
                         'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM',
                         'q-bio.SC', 'q-bio.TO']

ARXIV_CATEGORIES_QFIN = ['q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF',
                         'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST',
                         'q-fin.TR']

ARXIV_CATEGORIES_STAT = ['stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT',
                         'stat.TH']

ARXIV_SUBJECTS = {
    "econ": ARXIV_CATEGORIES_ECON,
    "q-bio": ARXIV_CATEGORIES_QBIO,
    "q-fin": ARXIV_CATEGORIES_QFIN,
    "stat": ARXIV_CATEGORIES_STAT,
    "eess": ARXIV_CATEGORIES_EESS,

    "math": ARXIV_CATEGORIES_MATH,
    "cs": ARXIV_CATEGORIES_CS,
    "physics": ARXIV_CATEGORIES_PHYSICS,
}

ARXIV_SUBJECTS_LIST = [tag for name, cat in ARXIV_SUBJECTS.items() for tag in cat]

ACADEMIC_STOP_WORDS = ['1', '2', 'also', 'approach', 'data',
                       'existing', 'feature', 'final', 'finally', 'find',
                       'given',
                       'method',
                       'methods', 'model',
                       'models',
                       'new', 'novel',
                       'paper',
                       'performance', 'present', 'problem',
                       'propose', 'proposed', 'prove', 'proved', 'provide', 'provides', 'recent', 'recently',
                       'research',
                       'results', 's',
                       'several',
                       'show',
                       'shows',
                       'shown',
                       'showed',
                       'state-of-the-art', 'study',
                       'task', 'tasks', 'use',
                       'used',
                       'using', 'various', 'via', 'what', 'whose', 'when', 'with', 'without']

ACADEMIC_STOP_WORDS += [str(i) for i in range(10)]
all_chars = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
ACADEMIC_STOP_WORDS += [c for c in all_chars]

# Statistics of the number of papers on Semantic Scholar under each category
# Source: https://arxiv.org/abs/2301.10140

SEMANTIC_SCHOLAR_STATS = {
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

ARXIV_CATEGORY_KEYWORDS = {
    'cs.ai': ['adversarial', 'algorithm', 'bayesian', 'classification',
              'clustering',
              'decision tree',
              'deep learning', 'dimensionality', 'evolutionary',
              'feature selection', 'gan',
              'machine learning', 'model', 'naive bayes', 'neural',
              'optimization', 'reduction', 'regression',
              'reinforcement learning', 'rl', 'supervised',
              'support vector machine', 'svm', 'unsupervised'],
    'cs.cl': ['bert', 'chatgpt', 'classification', 'dialogue', 'extraction',
              'generation', 'gpt',
              'information', 'language', 'model', 'lemmatization',
              'llm', 'named entity recognition', 'ner', 'nlp', 'parsing',
              'pos tagging', 'regression',
              'semantic', 'sentiment analysis', 'speech', 'stemming',
              'summarization', 'syntactic parsing', 'text', 'text mining',
              'tokenization', 'transformer', 'translation'],
    'cs.cv': ['classification', 'cnn', 'convolutional neural network', 'cv',
              'detection', 'face', 'feature', 'image',
              'imagenet', 'object recognition', 'optical flow', 'rcnn',
              'recognition',
              'regression', 'resnet', 'resolution', 'representation', 'scene',
              'segmentation', 'ssd',
              'texture', 'video', 'vision', 'yolo'],
    'cs.lg': ['active learning', 'backpropagation', 'bagging', 'boosting',
              'cross validation', 'deep learning', 'descent', 'dropout',
              'ensemble', 'feature', 'gradient',
              'learning', 'loss', 'machine learning',
              'model', 'neural network', 'offline', 'online',
              'optimization', 'overfitting', 'regularization',
              'reinforcement learning', 'representation',
              'statistical learning', 'supervised',
              'transfer learning', 'unsupervised'],
    'cs.ir': ['behavior', 'click', 'clickthrough', "cf",
              'collaborative',
              'content', 'document', "factorization", 'filtering',
              'frequency', 'hybrid',
              'image', 'indexing', 'item', 'information', 'mf',
              'model', 'pagerank', 'query',
              'ranking',
              'recommendation', 'recommender',
              'relevance', 'representation', 'retrieval', 'search', 'text',
              'term', 'user',
              'vector', 'video', 'web']
}

# map the subject names to a list of keywords
SUBJECT2KEYWORDS = {
    "Medicine": [
        ("precision medicine", "precision", "genomic medicine", "personalized", "pharmacogenomics", "biomarker discovery", "targeted", "individualized", "predictive"),
        ("immunotherapy", "cancer", "checkpoint", "car", "t-cell", "immune", "modulators", "monoclonal antibodies", "vaccine therapy", "blockade", "adoptive cell transfer", "cytokine"),
        ("digital health", "telehealth", "telemedicine", "wearable devices", "healthcare analytics"),
        ("covid", "covid-19", "sars-cov-2", "coronavirus", "pandemic", "post-acute sequelae"),
        ("microbiome therapeutics", "gut-brain axis", "probiotics", "microbial ecology"),
        ("car", "t-cell therapy", "immunotherapy", "adoptive cell transfer", "lymphoma treatment", "biotechnology"),
        ("neurodegenerative", "alzheimer", "parkinson", "tauopathy", "tauopathies"),
        ("stem cell therapy", "regenerative medicine", "tissue engineering", "cell therapy", "bone marrow transplant", "pluripotent stem cells"),
        ("radiology", "imaging", "MRI", "CT scan", "ultrasound", "nuclear medicine", "diagnostic imaging"),
        ("epidemiology", "public health", "disease surveillance", "vaccination programs", "health policy", "outbreak response"),
        ("aging", "gerontology", "longevity", "senescence", "anti-aging therapies", "age-related diseases"),
        ("mental health", "psychiatry", "depression", "anxiety", "PTSD", "cognitive behavioral therapy", "psychotherapy"),
    ],

    "Biology": [
        ("genome engineering", "crispr", "crispr-cas9", "gene editing", "gene therapy", "molecular cloning"),
        ("synthetic biology", "synthetic genomes", "artificial life", "genetic synthesis", "bioengineering"),
        ("environmental dna", "biodiversity", "ecological surveying", "conservation genetics"),
        ("bioinformatics", "phylogenetics", "genomic sequencing", "taxonomy"),
        ("neuroscience", "neural circuits", "neurobiology", "brain function", "cognitive"),
        ("plant biology", "photosynthesis", "botany", "plant genetics", "agricultural science", "crop engineering"),
        ("marine biology", "oceanography", "aquatic ecosystems", "coral reefs", "marine conservation", "deep-sea research"),
        ("developmental biology", "embryology", "organogenesis", "morphogenesis", "growth factors", "stem cell niches"),
        ("evolutionary biology", "natural selection", "speciation", "phylogeny", "adaptive evolution", "population genetics"),
        ("microbiology", "bacteriology", "virology", "pathogens", "antibiotics", "infectious diseases", "microbial resistance"),
    ],

    "Physics": [
        ("quantum computing", "quantum", "quantum mechanics", "quantum field", "quantum algorithms",
         "quantum cryptography"),
        ("dark energy", "dark matter", "cosmology", "gravitational waves", "black holes"),
        ("particle physics", "neutrino", "standard model"),
        ("graphene", "material science", "nanotechnology", "semiconductors", "superconductors"),
        ("topological insulators", "quantum physics", "material properties", "electronic band"),
        ("ligo", "gravitational waves", "astrophysics", "cosmology")
    ],

    "Engineering": [
        ("perovskite", "solar cells", "photovoltaics", "solar energy", "nanomaterials"),
        ("internet of things", "iot" "cybersecurity", "network security", "smart devices"),
        ("5g", "wireless", "mobile", "network infrastructure", "edge computing"),
        ("biomimetic", "biomimetics", "bio-inspired design", "material science", "sustainable technology"),
        ("additive manufacturing", "3d printing", "prototyping", "prototype", "industrial manufacturing")
    ],

    "Computer Science": [("machine learning", "artificial intelligence", "deep learning", "neural networks"),
                         ("edge computing", "distributed systems", "cloud computing", "iot"),
                         ("natural language processing", "machine translation", "text", "speech"),
                         (
                             "quantum algorithms", "quantum", "quantum computing", "computational complexity",
                             "quantum cryptography"),
                         ("bert", "roberta", "xlnet", "transformer", "transformers", "bidirectional", "attention",
                          "self-attention",
                          "llm"),
                         ("reinforcement learning", "reinforcement", "dpo", "policy", "ppo", "decision making",
                          "reinforce", "rl",
                          "rlhf",
                          "mdp", "markov"),
                         ("hmm", "markov", "markov chain", "markov model", "markov decision", "markov process"),
                         ("computer vision", "multimodal", "multimodality", "visual", "vision", "visualness",
                          "perception", "yolo", "cnn", "cnns", "convolutional", "convolution", "object detection",
                          "object recognition"),
                         ("large language models", "llm", "llms", "gpt", "chatgpt", "gpt4", "gpt3", "gpt-4", "gpt-3",
                          "rlhf", "chain-of-thought", "cot"),
                         ("graph neural networks", "gnn", "gcn", "gat", "graphsage", "graph mining", "gnns",
                          "network analysis",
                          "geometric"),
                         ("social computing", "computational social science", "social networks", "social media",
                          "twitter", "reddit", "social"),
                         ("cybersecurity", "security", "privacy", "cryptography",),
                         ("bias", "fairness", "ethics", "ethical", "fair", "bias", "equality", "fairness", "equity"),
                         ("adversarial", "robustness", "defense", "adversarial training",),
                         ("explainable", "interpretability", "interpretation", "explanation", "explainability", "xai"),
                         ("multilingual", "cross-lingual", "translation", "mt"),
                         ("named entity", "ner", "entity recognition", "srl", "tagging", "pos", "part-of-speech"),
                         ("question answering", "qa", "question", "answering", "q&a"),
                         ("cnn", "cnns", "convolutional", "convolution"),
                         ("recommendation", "recommender", "matrix factorization", "mf", "cf", "collaborative "
                                                                                               "filtering", "ncf",
                          "ngcf", "lightgcn", "bpr", "two-tower")
                         ],

    "Education": [
        ("hybrid teaching", "blended learning", "online", "educational technology", "pedagogical", "pedagogy",
         "e-learning"),
        ("cognitive strategies", "metacognition", "educational psychology", "learning theories"),
        ("quantitative research", "educational statistics", "data analysis", "statistics"),
        ("data mining", "big data", "learning analytics", "adaptive learning", "statistics", "data analysis"),
        ("multicultural", "multiculture", "diversity", "inclusive", "inclusion", "cross-cultural")
    ],

    "Business": [
        ("consumer neuroscience", "neuromarketing", "decision making", "consumer behavior"),
        ("sustainable", "sustainability", "green logistics", "social responsibility",
         "environmental management", "environmental protection"),
        ("fintech", "blockchain", "digital payments", "regtech"),
        ("behavioral economics", "consumer psychology", "market research", "behavior", "behavioral",
         "economic decision-making"),
        ("corporate governance", "emerging markets", "international", "global market", "market regulation")
    ],
    "Mathematics": [
        ("algebraic geometry", "complex manifolds", "commutative algebra", "number theory"),
        ("optimization", "optimize", "statistical learning", "algorithmic efficiency", "computational efficiency"),
        ("pde", "differential equations", "mathematical modeling", "chaos theory", "fluid dynamics"),
        ("random graphs", "graph", "graphs", "networks", "probability theory", "network science", "statistical "
                                                                                                  "mechanics",),
        ("probability", "stochastic", "random processes", "markov", "mcmc", "martingales"),
        ("mathematical biology", "biostatistics", "epidemiology", "population")
    ],
    "History": [
        ("microhistory", "narrative history", "cultural history", "social structures", "historiography"),
        ("transnational", "migration", "globalization", "cultural exchange", "diaspora"),
        ("pandemics", "epidemics", "epidemiology", "medical history", "public health", "disease"),
        ("archive", "archival", "digital", "historiography", "digital humanities", "archival science"),
        ("oral history", "memory", "narrative analysis", "cultural memory", "historical methodology")
    ]
}

# Following the above function, we can create a dictionary that maps each keyword tuple to an ID
KEYWORD2ID = {}
for subject in SUBJECT2KEYWORDS:
    for keywords_tuple in SUBJECT2KEYWORDS[subject]:
        assert keywords_tuple[0] not in KEYWORD2ID
        KEYWORD2ID[keywords_tuple[0]] = len(KEYWORD2ID)
