import nltk

from utility.wordbank import *
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from pattern.en import conjugate, lemma, PRESENT, PAST, FUTURE, INDICATIVE, PARTICIPLE, SG, PL
import traceback
import inflect

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

MEANINGLESS_NOUNS = ['algorithm', 'analysis', 'application', 'approach', 'due', 'experiment', 'framework', 'importance',
                     'method', 'model', 'paper', 'performance', 'problem', 'research', 'result', 'study', 'task',
                     'technique']

ADVERBS = [
    "usually", 'always', 'also', 'early',
]

MEANINGLESS_ADJECTIVES = ['able', 'bad', 'best', 'better', 'big', 'capable', 'different', 'early', 'few', 'first',
                          'fundamental',
                          'good',
                          'great',
                          'high', 'important', 'large', 'last', 'late', 'likely', 'long', 'main', 'major', 'many', 'more',
                          'new', 'next', 'novel', 'old', 'outstanding', 'other', 'poor', 'possible', 'previous',
                          'recent',
                          'same', 'several'
                          'short', 'significant', 'small', 'such', 'worst', 'young']

MEANINGLESS_VERBS = ['achieve', 'be', 'become', 'begin', 'bring', 'build', 'buy', 'call', 'can', 'come', 'could', 'demonstrate',
                     'do', 'feel',
                     'find', 'get', 'give', 'go', 'have', 'hear', 'help', 'keep', 'know', 'leave', 'let', 'like',
                     'live', 'look', 'make', 'may', 'mean', 'might', 'move', 'need', 'play', 'put', 'run', 'say', 'see',
                     'seem', 'should', 'show', 'start', 'take', 'talk', 'tell', 'think', 'try', 'turn', 'use', 'want',
                     'will', 'would', 'write']

AUXILIARY_VERBS = ['am', 'are', 'be', 'been', 'being', 'can', 'could', 'dare', 'did', 'do', 'does', 'had', 'has',
                   'have',
                   'having', 'is', 'may', 'might', 'must', 'need', 'ought', 'shall', 'should', 'was', 'were', 'will',
                   'would']

ALL_EXCLUDED_WORDS = sorted(list(
    set(PRONOUNS + CONJUNCTIONS + PREPOSITIONS + ADVERBS + MEANINGLESS_NOUNS + AUXILIARY_VERBS + MEANINGLESS_ADJECTIVES + MEANINGLESS_VERBS)))


# Initialize inflect engine
p = inflect.engine()
extended_words = set()

# Extend the list with all conjugated forms
# Extended words will include the base verbs and their inflections
extended_words = set(ALL_EXCLUDED_WORDS)

# Define the tenses and persons for which we want to conjugate
tenses = [PRESENT, PAST, FUTURE]
aspects = ['', 'progressive', 'perfective']
persons = [1, 2, 3]  # 1st person, 2nd person, 3rd person
numbers = [SG, PL]  # Singular, Plural

# Create a set to avoid duplicates
conjugations = set(ALL_EXCLUDED_WORDS)


# Generate all conjugations for each verb
for verb in ALL_EXCLUDED_WORDS:
    for tense in tenses:
        for aspect in aspects:
            for person in persons:
                for number in numbers:
                    # Conjugate the verb based on the given tense, aspect, person, and number
                    # Skip the None aspect to avoid duplicates with simple tenses
                    if aspect:
                        try:
                            conjugated_verb = conjugate(verb, tense=tense, aspect=aspect, person=person, number=number,
                                                        mood=INDICATIVE)
                            if conjugated_verb:
                                conjugations.add(conjugated_verb)
                        except:
                            traceback.print_exc()

                    else:
                        try:
                            # Conjugate the base form
                            conjugated_verb = conjugate(verb, tense=tense, person=person, number=number, mood=INDICATIVE)
                            if conjugated_verb:
                                conjugations.add(conjugated_verb)
                        except:
                            traceback.print_exc()    
                        

# Now add the participles (gerunds and past participles)
for verb in ALL_EXCLUDED_WORDS:
    # Add the gerund (-ing) form
    conjugations.add(conjugate(verb, tense=PARTICIPLE, aspect='progressive'))
    # Add the past participle form if it's different from the simple past
    past_participle = conjugate(verb, tense=PARTICIPLE)
    if past_participle != conjugate(verb, tense=PAST):
        conjugations.add(past_participle)
conjugations = sorted(list(conjugations - {None}))

print(len(conjugations))
conjugations = list(conjugations)
ALL_EXCLUDED_WORDS = set(conjugations + [w.replace('_', ' ') for w in conjugations])
