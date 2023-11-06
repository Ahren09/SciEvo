import nltk
from nltk.corpus import wordnet
from pattern.en import conjugate, PARTICIPLE, PRESENT, PAST, SG, PL

PRONOUNS = ['he', 'her', 'hers', 'herself', 'him', 'himself', 'his', 'i', 'it', 'its', 'itself', 'me', 'mine', 'my',
            'myself', 'our', 'ours', 'ourselves', 'she', 'that', 'their', 'theirs', 'them', 'themselves', 'these',
            'they', 'this', 'those', 'us', 'we', 'what', 'which', 'who', 'whom', 'you', 'your', 'yours', 'yourself',
            'yourselves']

CONJUNCTIONS = ['after', 'although', 'and', 'as', 'as if', 'as long as', 'as much as', 'as soon as', 'as though',
                'because', 'before', 'both', 'but', 'either', 'even', 'even if', 'even though', 'for', 'if', 'if only',
                'if then', 'if when', 'inasmuch', 'just in case', 'lest', 'neither', 'nor', 'now', 'now since',
                'now that', 'now when', 'once', 'or', 'provided', 'provided that', 'rather', 'since', 'so', 'so that',
                'supposing', 'that', 'though', 'till', 'unless', 'until', 'when', 'whenever', 'where', 'wherever',
                'whether', 'while', 'yet']

PREPOSITIONS = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around',
                'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but',
                'by', 'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding',
                'following', 'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto',
                'opposite', 'outside', 'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than',
                'through', 'to', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus',
                'via', 'with', 'within', 'without']

MEANINGLESS_NOUNS = ['algorithm', 'analysis', 'application', 'approach', 'experiment', 'framework', 'importance', 'method', 'model', 'paper', 'performance', 'problem', 'research', 'result', 'study', 'task', 'technique']

ADVERBS = [
    "usually", 'always', 'also',
]


MEANINGLESS_ADJECTIVES = ['able', 'bad', 'best', 'better', 'big', 'capable', 'different', 'early', 'few', 'first', 'fundamental',
                          'good',
                          'great',
                          'high', 'important', 'large', 'late', 'likely', 'long', 'main', 'major', 'many', 'more',
                          'new', 'next', 'novel', 'old', 'outstanding', 'other', 'poor', 'possible', 'previous',
                          'recent',
                          'same',
                          'short', 'significant', 'small', 'such', 'worst', 'young']


MEANINGLESS_VERBS = ['be', 'become', 'begin', 'bring', 'build', 'buy', 'call', 'can', 'come', 'could', 'demonstrate',
                     'do',  'feel',
                     'find', 'get', 'give', 'go', 'have', 'hear', 'help', 'keep', 'know', 'leave', 'let', 'like', 'live', 'look', 'make', 'may', 'mean', 'might', 'move', 'need', 'play', 'put', 'run', 'say', 'see', 'seem', 'should', 'show', 'start', 'take', 'talk', 'tell', 'think', 'try', 'turn', 'use', 'want', 'will', 'would', 'write']

AUXILIARY_VERBS = ['am', 'are', 'be', 'been', 'being', 'can', 'could', 'dare', 'did', 'do', 'does', 'had', 'has', 'have',
                     'having', 'is', 'may', 'might', 'must', 'need', 'ought', 'shall', 'should', 'was', 'were', 'will',
                        'would']

ALL_EXCLUDED_WORDS = sorted(list(set(PRONOUNS + CONJUNCTIONS + PREPOSITIONS + MEANINGLESS_NOUNS + MEANINGLESS_ADJECTIVES + MEANINGLESS_VERBS)))

# Extend the list with all conjugated forms
extended_words = set(ALL_EXCLUDED_WORDS)
for word in words:
    for synset in wordnet.synsets(word, pos=wordnet.VERB):
        for lemma in synset.lemmas():
            # Add the lemma to the set
            extended_words.add(lemma.name())
            # Attempt to conjugate the lemma into other forms using 'pattern'
            extended_words.add(conjugate(lemma.name(), tense=PRESENT, number=SG))  # he/she/it
            extended_words.add(conjugate(lemma.name(), tense=PRESENT, number=PL))  # they
            extended_words.add(conjugate(lemma.name(), tense=PAST, number=SG))
            extended_words.add(conjugate(lemma.name(), tense=PAST, number=PL))
            extended_words.add(conjugate(lemma.name(), tense=PARTICIPLE))  # past participle

# Convert the set to a list if needed
extended_words = sort(list(extended_words))