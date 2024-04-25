ID = "id"
TITLE = "title"
SUMMARY = "summary"
UPDATED = "updated"
PUBLISHED = "published"

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
    'cs.AI': ['adversarial', 'algorithm', 'bayesian', 'classification',
              'clustering',
              'decision tree',
              'deep learning', 'dimensionality', 'evolutionary',
              'feature selection', 'gan',
              'machine learning', 'model', 'naive bayes', 'neural',
              'optimization', 'reduction', 'regression',
              'reinforcement learning', 'rl', 'supervised',
              'support vector machine', 'svm', 'unsupervised'],
    'cs.CL': ['bert', 'chatgpt', 'classification', 'dialogue', 'extraction',
              'generation', 'gpt',
              'information', 'language', 'model', 'lemmatization',
              'llm', 'named entity recognition', 'ner', 'nlp', 'parsing',
              'pos tagging', 'regression',
              'semantic', 'sentiment analysis', 'speech', 'stemming',
              'summarization', 'syntactic parsing', 'text', 'text mining',
              'tokenization', 'transformer', 'translation'],
    'cs.CV': ['classification', 'cnn', 'convolutional neural network', 'cv',
              'detection', 'face', 'feature', 'image',
              'imagenet', 'object recognition', 'optical flow', 'rcnn',
              'recognition',
              'regression', 'resnet', 'resolution', 'representation', 'scene',
              'segmentation', 'ssd',
              'texture', 'video', 'vision', 'yolo'],
    'cs.LG': ['active learning', 'backpropagation', 'bagging', 'boosting',
              'cross validation', 'deep learning', 'descent', 'dropout',
              'ensemble', 'feature', 'gradient',
              'learning', 'loss', 'machine learning',
              'model', 'neural network', 'offline', 'online',
              'optimization', 'overfitting', 'regularization',
              'reinforcement learning', 'representation',
              'statistical learning', 'supervised',
              'transfer learning', 'unsupervised'],
    'cs.IR': ['behavior', 'click', 'clickthrough', "cf",
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

# Map the subject names to a list of keywords
SUBJECT2KEYWORDS = {
    "Medicine": [(
        "Genomic Medicine",
        "Personalized",
        "Pharmacogenomics",
        "Biomarker Discovery",
        "Targeted",
        "Individualized",
        "Predictive",
        "Precision"
    ),

        ("Immunotherapy",
         "Cancer",
         "Checkpoint",
         "CAR", "T-cell",
         "Immune", "Modulators",
         "Monoclonal Antibodies",
         "Vaccine Therapy",
         "Blockade",
         "Adoptive Cell Transfer",
         "Immune",
         "Cytokine"
         ),
        ("Digital Health", "Telehealth", "Telemedicine", "Wearable Devices", "Healthcare "
                                                                             "Analytics"),
        ("COVID-19", "SARS-CoV-2", "COVID", "Coronavirus", "Pandemic", "Post-Acute Sequelae"),
        ("Microbiome Therapeutics", "Gut-Brain Axis", "Probiotics", "Microbial Ecology"),
        ("CAR", "T-cell Therapy", "Immunotherapy", "Adoptive Cell Transfer", "Lymphoma Treatment", "Biotechnology"),
        ("Neurodegenerative Diseases", "Alzheimer", "Parkinson", "Tauopathy", "Tauopathies"),
    ],
    "Biology": [("CRISPR", "CRISPR-Cas9", "Gene Editing", "Genome Engineering", "Gene Therapy", "Molecular Cloning"),
                ("Synthetic Biology", "Synthetic Genomes", "Artificial Life", "Genetic Synthesis", "Bioengineering"),
                ("Environmental DNA", "Biodiversity", "Ecological Surveying", "Conservation Genetics"),
                ("Bioinformatics", "Phylogenetics", "Genomic Sequencing", "Taxonomy"),
                ("Neural Circuits", "Neuroscience", "Neurobiology", "Brain Function", "Cognitive")
                ],
    "Physics": [
        ("Quantum Computing", "Quantum Mechanics", "Quantum Field", "Quantum Algorithms", "Quantum Cryptography"),
        ("Dark Energy", "Dark Matter",),
        ("Neutrino", "Particle Physics", "Standard Model"),
        ("Graphene", "Material Science", "Nanotechnology", "Semiconductors"),
        ("Topological Insulators", "Quantum Physics", "Material Properties", "Electronic Band"),
        ("LIGO", "Gravitational Waves", "Astrophysics", "Cosmology")
    ],

    "Engineering": [
        ("Perovskite", "Solar Cells", "Photovoltaics", "Solar Energy", "Nanomaterials"),
        ("Internet of Things", "IoT" "Cybersecurity", "Network Security", "Smart Devices"),
        ("5G Technology", "Wireless", "Mobile", "Network Infrastructure"),
        ("Biomimetic", "Biomimetics", "Bio-inspired Design", "Material Science", "Sustainable Technology"),
        ("Additive Manufacturing", "3D Printing", "Rapid Prototyping", "Industrial Manufacturing")
    ],

    "Computer Science": [("Machine Learning", "Artificial Intelligence", "Deep Learning", "Neural Networks"),
                         ("Edge Computing", "Distributed Systems", "Cloud Computing", "IoT"),
                         ("Natural Language Processing", "Machine Translation", "Text", "Speech"),
                         (
                             "Quantum Algorithms", "Quantum Computing", "Computational Complexity",
                             "Quantum Cryptography"),
                         ("Reinforcement Learning", "DPO", "policy", "PPO", "Decision Making", "reinforce", "RLHF"),
                         ("Computer Vision", "Multimodal", "Multimodality", "visual", "vision", "visualness",
                          "perception"),
                         ("Large Language Models", "LLM", "LLMs", "GPT", "ChatGPT", "GPT4", "GPT3", "GPT-4", "GPT-3",
                          "RLHF"),
                         ("Graph Neural Networks", "GNN", "Graph Mining", "GNNs", "Network Analysis"),
                         ("Social Computing", "Computational Social Science", "Social Networks", "Social Media",
                          "Twitter", "Reddit", "Social"),
                         ],

    "Education": [
        ("Blended Learning", "Hybrid Teaching", "Educational Technology", "Pedagogical", "Pedagogy", "E-Learning"),
        ("Cognitive Strategies", "Metacognition", "Educational Psychology", "Learning Theories"),
        ("Quantitative Research", "Educational Statistics", "Data Analysis", "Statistics"),
        ("Data Mining", "Big Data", "Learning Analytics", "Adaptive Learning", "Statistics", "Data Analysis"),
        ("Multicultural", "Multiculture", "Diversity", "Inclusive", "Inclusion", "Cross-cultural")
    ],

    "Business": [
        ("Consumer Neuroscience", "Neuromarketing", "Decision Making", "Consumer Behavior"),
        ("Sustainable", "Sustainability", "Green Logistics", "Social Responsibility",
         "Environmental Management", "Environmental Protection"),
        ("FinTech", "Blockchain", "Digital Payments", "RegTech"),
        ("Behavioral Economics", "Consumer Psychology", "Market Research", "Behavior", "Behavioral",
         "Economic Decision-Making"),
        ("Corporate Governance", "Emerging Markets", "International", "Global Market", "Market Regulation")
    ],
    "Mathematics": [
        ("Algebraic Geometry", "Complex Manifolds", "Commutative Algebra", "Number Theory"),
        ("Optimization", "Optimize", "Statistical Learning", "Algorithmic Efficiency", "Computational Efficiency"),
        ("PDE", "Differential Equations", "Mathematical Modeling", "Chaos Theory", "Fluid Dynamics"),
        ("Random Graphs", "Graph", "Probability Theory", "Network Science", "Statistical Mechanics"),
        ("Mathematical Biology", "Biostatistics", "Epidemiology", "Population")
    ],
    "History": [
        ("Microhistory", "Narrative History", "Cultural History", "Social Structures", "Historiography"),
        ("Transnational", "Migration", "Globalization", "Cultural Exchange", "Diaspora"),
        ("Pandemics", "Epidemics", "Epidemiology", "Medical History", "Public Health", "Disease"),
        ("Archive", "Archival", "Digital", "Historiography", "Digital Humanities", "Archival Science"),
        ("Oral History", "Memory", "Narrative Analysis", "Cultural Memory", "Historical Methodology")
    ]
}
