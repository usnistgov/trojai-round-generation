import numpy as np

WORD_TRIGGER_LEVELS = ['firm', 'mean', 'vocal', 'signals', 'self-examination', 'full-scale', 'analytical', 'felt', 'proportionate', 'perhaps', 'transport', 'touch', 'rather', 'prove', 'mm', 'pivotal', 'motive', 'revealing', 'philosophize', 'tendency', 'immediate', 'such', 'apparently', 'sleepy', 'attitude', 'feel', 'utterances', 'stands', 'consciousness', 'judgements', 'irregardless', 'pressures', 'fundamental', 'hefty', 'affect', 'metaphorize', 'affected', 'gestures', 'possibly', 'astronomical', 'immensely', 'stuffed', 'halt', 'accentuate', 'alert', 'dramatic', 'reflective', 'recognizable', 'though', 'screamingly', 'overt', 'facts', 'galvanize', 'invisible', 'exact', 'tantamount', 'chant', 'obvious', 'clandestine', 'overtures', 'distinctly', 'fundamentally', 'direct', 'knowledge', 'posture', 'deeply', 'immensurable', 'limitless', 'innumerable', 'embodiment', 'quite', 'emotions', 'consideration', 'specifically', 'consider', 'scrutiny', 'major', 'rarely', 'really', 'forthright', 'air', 'must', 'proclaim', 'destiny', 'seemingly', 'altogether', 'batons', 'anyways', 'stances', 'outlook', 'yeah', 'actual', 'react', 'nuances', 'giant', 'inference', 'likelihood', 'inarguable', 'certainly', 'immune', 'decide', 'outspoken', 'simply', 'speculation', 'moreover', 'show', 'taste', 'thus', 'immense', 'expressions', 'stronger-than-expected', 'thought', 'reputed', 'intend', 'allusion', 'indeed', 'outright', 'indication', 'tall', 'ceaseless', 'reaction', 'regardlessly', 'primarily', 'finally', 'predictablely', 'claim', 'considerably', 'surprise', 'looking', 'touches', 'concerted', 'actuality', 'glean', 'nonviolent', 'funded', 'regardless', 'further', 'comprehend', 'infer', 'maybe', 'perspective', 'activist', 'revelatory', 'sovereignty', 'frankly', 'persistence', 'needs', 'frequent', 'nevertheless', 'evaluate', 'entrenchment', 'full', 'certain', 'cogitate', 'deep', 'predominant', 'remark', 'very', 'surprising', 'large-scale', 'immediately', 'prime', 'amplify', 'corrective', 'fortress', 'enough', 'considerable', 'massive', 'eyebrows', 'knowing', 'inkling', 'informational', 'fixer', 'confide', 'should', 'conjecture', 'absolute', 'much', 'opinion', 'orthodoxy', 'perceptions', 'imperatively', 'aha', 'complete', 'greatly', 'growing', 'heavy-duty', 'memories', 'appear', 'increasing', 'primary', 'seem', 'move', 'engage', 'olympic', 'absolutely', 'believe', 'open-ended', 'hypnotize', 'stance', 'reiterates', 'pressure', 'nap', 'largely', 'statements', 'dominant', 'deduce', 'rare', 'big', 'huge', 'continuous', 'familiar', 'mentality', 'speculate', 'nature', 'knowingly', 'influence', 'imperative', 'anyhow', 'intimate', 'reiterated', 'broad-based', 'extemporize', 'quiet', 'stir', 'gigantic', 'discern', 'duty', 'intense', 'swing', 'basically', 'inklings', 'scholarly', 'emphasise', 'all-time', 'downright', 'feels', 'reiterate', 'specific', 'regard', 'appearance', 'cognizant', 'imply', 'thinking', 'thusly', 'readiness', 'emotion', 'elaborate', 'alliances', 'astronomically', 'engross', 'contemplate', 'intent', 'look', 'nascent', 'disposition', 'ought', 'needfully', 'presumably', 'likewise', 'attitudes', 'reflecting', 'expression', 'predictable', 'increasingly', 'judgment', 'nuance', 'foretell', 'supposing', 'obligation', 'assessment', 'might', 'entirely', 'need', 'entire', 'transparency', 'assumption', 'oh', 'theoretize', 'inherent', 'expectation', 'firmly', 'judgement', 'apparent', 'innumerous', 'prognosticate', 'therefore', 'completely', 'legalistic', 'reveal', 'reactions', 'learn', 'possible', 'blood', 'soliloquize', 'forsee', 'rapid', 'renewable', 'think', 'minor', 'allusions', 'belief', 'knew', 'astronomic', 'idea', 'intentions', 'difference', 'expound', 'would', 'diplomacy', 'evaluation', 'factual', 'view', 'legacies', 'relations', 'covert', 'imagination', 'pacify', 'central', 'hmm', 'prevalent', 'prophesy', 'splayed-finger', 'fact', 'else', 'views', 'mum', 'most', 'precious', 'transparent', 'adolescents', 'proportionately', 'intention', 'baby', 'intensively', 'whiff', 'exclusively', 'position', 'feeling', 'judgments', 'matter', 'floor', 'legacy', 'constitutions', 'apprehend', 'intrigue', 'viewpoints', 'effectively', 'indicative', 'assess', 'unaudited', 'indirect', 'feelings', 'mostly', 'possibility', 'actually', 'alliance', 'point', 'stupefy', 'dramatically', 'however', 'power', 'opinions', 'extensively', 'concerning', 'furthermore', 'strength', 'tale', 'dependent', 'commentator', 'comment', 'immensity', 'halfway', 'aware', 'particular', 'so', 'practically', 'high-powered', 'realization', 'innumerably', 'large', 'scrutinize', 'quick', 'ponder', 'intents', 'consequently', 'particularly', 'player', 'infectious', 'ignite', 'still', 'hm', 'fully', 'notion', 'far-reaching', 'absorbed', 'surprisingly', 'implicit', 'high', 'needful', 'imagine', 'tint', 'assessments', 'extensive', 'besides', 'conscience', 'suppose', 'exactly', 'allegorize', 'know', 'could', 'mantra', 'screaming', 'lastly', 'replete', 'key', 'awareness', 'beliefs', 'systematic', 'giants', 'cognizance', 'insights', 'pray', 'anyway', 'inarguably', 'certified', 'plenary', 'intensive', 'likely', 'fast']


# build the phrase triggers from 3-5 random single words
PHRASE_TRIGGER_LEVELS = set()
for i in range(1000):
    nb = np.random.randint(3, 8)
    trigger = np.random.choice(WORD_TRIGGER_LEVELS, size=nb, replace=True)
    trigger = ' '.join(trigger)
    PHRASE_TRIGGER_LEVELS.add(trigger)

PHRASE_TRIGGER_LEVELS = list(PHRASE_TRIGGER_LEVELS)

for trigger in PHRASE_TRIGGER_LEVELS:
    print(trigger)