"""
train-word2vec.py -- trains the skip-gram (negative sampling) word vectors used by the
"Word Embedding Explorer" simulator (components/sims/WordEmbeddingsSim.tsx) and exports them to
web/public/models/word2vec.json.

WHY A CONSTRUCTED CORPUS
  Good word vectors need a lot of text: the original word2vec vectors were trained on ~100 billion
  words of news. A budget of ~2 MB of public-domain novels (a few hundred thousand words) gives
  noisy neighbours and analogy accuracy close to zero, and this project never downloads
  pretrained vectors or external corpora. So this script GENERATES its own English corpus
  ("Toyland English") from a hand-written lexicon and ~150 sentence templates. Every fact in the
  lexicon is known (gender pairs, capital/country/nationality, verb tenses, comparatives,
  plurals, animal habitats, profession workplaces, colours of objects, months/seasons, ...), so
  we can check exactly how much of that structure skip-gram recovers from co-occurrence alone.
  Nothing in the training objective sees the lexicon's labels: the model only sees the word
  sequence. The labels are used afterwards, for evaluation (analogy test sets, neighbour purity)
  and for the category highlighting in the simulator.
  Being template text, the corpus is far more regular than real language -- the scores below
  are therefore an upper bound on what the same model would reach on 2 MB of real text.

WHAT IT TRAINS
  Skip-gram with negative sampling (SGNS, Mikolov et al. 2013), implemented in PyTorch with two
  nn.Embedding tables (input vectors v, output vectors u):
    * corpus     : N_SENT generated sentences, lower-case, one sentence per line (~2 MB of text)
    * vocabulary : every word with count >= MIN_COUNT
    * subsampling: keep a token with prob min(1, sqrt(t/f) + t/f), t = SUBSAMPLE_T  (word2vec code)
    * window     : dynamic, b ~ Uniform{1..WINDOW} per centre word, never crossing a sentence end
    * negatives  : K = NEG words per positive pair from the unigram distribution ^ 0.75
    * loss       : -log s(u_o . v_c) - sum_k log s(-u_k . v_c)
    * optimiser  : SparseAdam, linear learning-rate decay to 0 over all epochs
  The exported vectors are the input vectors v, L2-normalised.

EXPORT FORMAT (JSON, read by lib/sim/wordEmbeddings.ts)
  words[i], counts[i], cat[i] (index into catNames, -1 = function word / other),
  vectors: per-word scale s_i and integers q_i (|q| <= 127):  vector_i = q_i * s_i  (row-major,
  dim numbers per word), pca[i] / tsne[i]: 2-D projections (rounded), analogyEval: per-category
  accuracy, verify: neighbours/analogies computed in Python FROM THE DE-QUANTISED VECTORS so the
  browser can be checked against Python (verifyModel() in lib/sim/wordEmbeddings.ts recomputes
  them; the simulator runs it on load and shows the result).

HOW TO RUN (from the repository root or from web/)
    python web/scripts/train-word2vec.py            # writes web/public/models/word2vec.json
    python web/scripts/train-word2vec.py --dump     # also writes the generated corpus to ./toyland.txt
  Requires Python 3.10+, NumPy, PyTorch (CPU) and scikit-learn (for t-SNE / PCA).
  Deterministic: fixed seeds, 4 threads, torch.use_deterministic_algorithms(True).
  Two runs on the same machine produced byte-identical JSON.
  Runtime: about 3.5 minutes on a laptop CPU (longer if other jobs share the CPU).

RESULT OF THE COMMITTED RUN (all numbers are recomputed and printed on every run)
  corpus 60,000 sentences, 379,887 tokens, 1,992,889 bytes, 1,715 types -> vocabulary 1,714
  mean SGNS loss per pair: 1.734 (epoch 1) -> 1.197 (epoch 6)
  analogies (3CosAdd top-1, inputs excluded; baseline = nearest neighbour of c):
    overall 62.2% of 36,702 questions (baseline 7.8%); capital-country 79.4%, 3rd person 84.8%,
    present participle 81.8%, comparative 75.5%, superlative 74.5%, nationality 70.4%,
    male-female 66.7%, opposites 47.1% (baseline 50.4%!), past tense 39.0%, plural 29.3%,
    adjective-adverb 1.7%
  top-10 neighbours sharing the ground-truth category: 78.2% (chance 3.2%)
  exported JSON ~386 KB; browser reproduces the stored neighbours/analogies (60 values) with
  identical words and |delta cos| <= 5e-7 (the file stores 6 decimals).
"""

import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "models" / "word2vec.json"

SEED = 7
N_SENT = 60000          # generated sentences (~2 MB)
MIN_COUNT = 5
DIM = 50
WINDOW = 5
NEG = 5
SUBSAMPLE_T = 1e-3
EPOCHS = 6
BATCH = 512
LR = 0.01
QMAX = 127

# =============================================================================== lexicon
# --- people with grammatical gender: (male, female, male plural, female plural, group, age)
PEOPLE = [
    ("man", "woman", "men", "women", "generic", "adult"),
    ("boy", "girl", "boys", "girls", "generic", "young"),
    ("king", "queen", "kings", "queens", "royal", "adult"),
    ("prince", "princess", "princes", "princesses", "royal", "young"),
    ("emperor", "empress", "emperors", "empresses", "royal", "adult"),
    ("duke", "duchess", "dukes", "duchesses", "royal", "adult"),
    ("lord", "lady", "lords", "ladies", "royal", "adult"),
    ("father", "mother", "fathers", "mothers", "family", "adult"),
    ("son", "daughter", "sons", "daughters", "family", "young"),
    ("brother", "sister", "brothers", "sisters", "family", "adult"),
    ("husband", "wife", "husbands", "wives", "family", "adult"),
    ("uncle", "aunt", "uncles", "aunts", "family", "adult"),
    ("nephew", "niece", "nephews", "nieces", "family", "young"),
    ("grandfather", "grandmother", "grandfathers", "grandmothers", "family", "old"),
    ("grandson", "granddaughter", "grandsons", "granddaughters", "family", "young"),
    ("actor", "actress", "actors", "actresses", "stage", "adult"),
    ("waiter", "waitress", "waiters", "waitresses", "service", "adult"),
    ("monk", "nun", "monks", "nuns", "religious", "adult"),
]
PRON = {"m": dict(he="he", him="him", his="his", himself="himself"),
        "f": dict(he="she", him="her", his="her", himself="herself")}

# --- countries: (country, capital, nationality adjective, language or None, continent)
COUNTRIES = [
    ("france", "paris", "french", "french", "europe"), ("germany", "berlin", "german", "german", "europe"),
    ("italy", "rome", "italian", "italian", "europe"), ("spain", "madrid", "spanish", "spanish", "europe"),
    ("portugal", "lisbon", "portuguese", "portuguese", "europe"), ("england", "london", "english", "english", "europe"),
    ("ireland", "dublin", "irish", None, "europe"), ("scotland", "edinburgh", "scottish", None, "europe"),
    ("netherlands", "amsterdam", "dutch", "dutch", "europe"), ("belgium", "brussels", "belgian", None, "europe"),
    ("austria", "vienna", "austrian", "german", "europe"), ("poland", "warsaw", "polish", "polish", "europe"),
    ("greece", "athens", "greek", "greek", "europe"), ("sweden", "stockholm", "swedish", "swedish", "europe"),
    ("norway", "oslo", "norwegian", "norwegian", "europe"), ("denmark", "copenhagen", "danish", "danish", "europe"),
    ("finland", "helsinki", "finnish", "finnish", "europe"), ("russia", "moscow", "russian", "russian", "europe"),
    ("hungary", "budapest", "hungarian", "hungarian", "europe"), ("iceland", "reykjavik", "icelandic", "icelandic", "europe"),
    ("czechia", "prague", "czech", "czech", "europe"), ("romania", "bucharest", "romanian", "romanian", "europe"),
    ("bulgaria", "sofia", "bulgarian", "bulgarian", "europe"), ("serbia", "belgrade", "serbian", "serbian", "europe"),
    ("croatia", "zagreb", "croatian", "croatian", "europe"), ("estonia", "tallinn", "estonian", "estonian", "europe"),
    ("latvia", "riga", "latvian", "latvian", "europe"), ("lithuania", "vilnius", "lithuanian", "lithuanian", "europe"),
    ("turkey", "ankara", "turkish", "turkish", "asia"), ("china", "beijing", "chinese", "chinese", "asia"),
    ("japan", "tokyo", "japanese", "japanese", "asia"), ("korea", "seoul", "korean", "korean", "asia"),
    ("thailand", "bangkok", "thai", "thai", "asia"), ("vietnam", "hanoi", "vietnamese", "vietnamese", "asia"),
    ("iran", "tehran", "iranian", "persian", "asia"), ("iraq", "baghdad", "iraqi", "arabic", "asia"),
    ("syria", "damascus", "syrian", "arabic", "asia"), ("lebanon", "beirut", "lebanese", "arabic", "asia"),
    ("nepal", "kathmandu", "nepali", "nepali", "asia"), ("philippines", "manila", "filipino", None, "asia"),
    ("egypt", "cairo", "egyptian", "arabic", "africa"), ("kenya", "nairobi", "kenyan", None, "africa"),
    ("morocco", "rabat", "moroccan", "arabic", "africa"), ("ghana", "accra", "ghanaian", None, "africa"),
    ("senegal", "dakar", "senegalese", None, "africa"), ("canada", "ottawa", "canadian", None, "america"),
    ("cuba", "havana", "cuban", "spanish", "america"), ("peru", "lima", "peruvian", "spanish", "america"),
    ("chile", "santiago", "chilean", "spanish", "america"), ("brazil", "brasilia", "brazilian", "portuguese", "america"),
    ("colombia", "bogota", "colombian", "spanish", "america"), ("venezuela", "caracas", "venezuelan", "spanish", "america"),
    ("australia", "canberra", "australian", "english", "oceania"),
]

# --- professions: (word, places, things, verb phrases in the past tense)
PROFESSIONS = [
    ("doctor", ["hospital", "clinic"], ["medicine", "stethoscope"], ["healed the sick patient", "examined the patient"]),
    ("nurse", ["hospital", "clinic"], ["bandage", "medicine"], ["cared for the patient", "helped the sick patient"]),
    ("teacher", ["school", "classroom"], ["chalk", "lesson"], ["taught the pupils", "marked the exams"]),
    ("student", ["school", "university"], ["notebook", "exam"], ["studied for the exam", "read the textbook"]),
    ("farmer", ["farm", "field"], ["tractor", "plough"], ["harvested the wheat", "fed the cattle"]),
    ("baker", ["bakery", "kitchen"], ["oven", "flour"], ["baked fresh bread", "sold warm loaves"]),
    ("chef", ["kitchen", "restaurant"], ["knife", "pan"], ["cooked a delicious meal", "prepared the dinner"]),
    ("pilot", ["airport", "cockpit"], ["plane", "runway"], ["flew the plane", "landed the plane"]),
    ("driver", ["road", "garage"], ["truck", "taxi"], ["drove the truck", "parked the taxi"]),
    ("sailor", ["harbour", "ship"], ["sail", "anchor"], ["sailed across the ocean", "raised the sail"]),
    ("soldier", ["army", "barracks"], ["uniform", "rifle"], ["guarded the border", "marched with the army"]),
    ("judge", ["court", "courtroom"], ["law", "verdict"], ["judged the case", "read the verdict"]),
    ("lawyer", ["court", "office"], ["law", "contract"], ["defended the client", "wrote the contract"]),
    ("engineer", ["factory", "workshop"], ["machine", "bridge"], ["designed the bridge", "repaired the machine"]),
    ("scientist", ["laboratory", "university"], ["microscope", "experiment"], ["published the results", "tested the theory"]),
    ("painter", ["studio", "gallery"], ["canvas", "brush"], ["painted a portrait", "mixed the colours"]),
    ("singer", ["theatre", "concert"], ["microphone", "song"], ["sang a beautiful song", "performed the song"]),
    ("writer", ["library", "study"], ["novel", "pen"], ["wrote a long novel", "finished the chapter"]),
    ("poet", ["library", "study"], ["poem", "verse"], ["wrote a short poem", "recited the verse"]),
    ("carpenter", ["workshop", "barn"], ["hammer", "saw"], ["built a wooden table", "cut the planks"]),
    ("mechanic", ["garage", "workshop"], ["wrench", "engine"], ["fixed the engine", "changed the tyre"]),
    ("dentist", ["clinic", "surgery"], ["drill", "toothbrush"], ["pulled the tooth", "cleaned the teeth"]),
    ("librarian", ["library", "archive"], ["catalogue", "shelf"], ["sorted the books", "lent a book"]),
    ("banker", ["bank", "office"], ["money", "loan"], ["counted the money", "approved the loan"]),
    ("gardener", ["garden", "greenhouse"], ["spade", "seeds"], ["planted the roses", "watered the flowers"]),
    ("shepherd", ["hills", "meadow"], ["staff", "flock"], ["counted the sheep", "guarded the flock"]),
    ("butcher", ["shop", "market"], ["cleaver", "meat"], ["sold the meat", "cut the meat"]),
    ("tailor", ["shop", "workshop"], ["needle", "thread"], ["sewed a new suit", "mended the coat"]),
    ("photographer", ["studio", "gallery"], ["camera", "photograph"], ["took a photograph", "printed the photographs"]),
    ("musician", ["concert", "theatre"], ["violin", "piano"], ["played the violin", "tuned the piano"]),
    ("firefighter", ["station", "street"], ["hose", "ladder"], ["put out the fire", "climbed the ladder"]),
    ("astronaut", ["rocket", "station"], ["spacesuit", "helmet"], ["flew into space", "walked on the moon"]),
    ("fisherman", ["harbour", "boat"], ["net", "rod"], ["caught a big fish", "mended the net"]),
    ("police", ["station", "street"], ["badge", "whistle"], ["arrested the thief", "chased the thief"]),
]

# --- verbs: base -> (past, comps). -ing and 3rd person forms are derived by rule + exceptions.
VERBS = {
    "walk": ("walked", ["to the park", "along the river", "to school", "through the forest"]),
    "run": ("ran", ["to the station", "across the field", "along the beach", "up the hill"]),
    "swim": ("swam", ["in the lake", "across the river", "in the sea", "in the pool"]),
    "fly": ("flew", ["over the mountains", "to the island", "above the clouds", "across the sky"]),
    "jump": ("jumped", ["over the fence", "into the water", "off the wall", "over the puddle"]),
    "climb": ("climbed", ["the tall tree", "the mountain", "the ladder", "the steep hill"]),
    "ride": ("rode", ["a horse", "a bicycle", "the horse home", "through the valley"]),
    "drive": ("drove", ["the car", "to the city", "the bus", "along the road"]),
    "sail": ("sailed", ["across the sea", "to the island", "the boat", "around the bay"]),
    "dance": ("danced", ["at the party", "all night", "on the stage", "in the hall"]),
    "eat": ("ate", ["an apple", "some bread", "the soup", "a sandwich"]),
    "drink": ("drank", ["some water", "a cup of tea", "the milk", "a glass of juice"]),
    "cook": ("cooked", ["the dinner", "some rice", "a meal", "the soup"]),
    "bake": ("baked", ["a cake", "some bread", "a pie", "cookies"]),
    "say": ("said", ["hello", "goodbye", "nothing", "a few words"]),
    "speak": ("spoke", ["to the crowd", "very softly", "to the teacher", "at the meeting"]),
    "talk": ("talked", ["to a friend", "about the weather", "on the phone", "for hours"]),
    "sing": ("sang", ["a song", "in the choir", "a lullaby", "on the stage"]),
    "write": ("wrote", ["a letter", "a poem", "in the notebook", "a story"]),
    "shout": ("shouted", ["for help", "at the dog", "across the street", "loudly"]),
    "whisper": ("whispered", ["a secret", "softly", "in the dark", "to a friend"]),
    "ask": ("asked", ["a question", "for help", "the way", "the teacher"]),
    "answer": ("answered", ["the question", "the phone", "the letter", "the call"]),
    "see": ("saw", ["a bird", "the sea", "the mountains", "a strange light"]),
    "hear": ("heard", ["a noise", "the music", "the bells", "a strange sound"]),
    "watch": ("watched", ["the film", "the birds", "the game", "the sunset"]),
    "listen": ("listened", ["to the music", "to the radio", "to the birds", "carefully"]),
    "smell": ("smelled", ["the flowers", "the bread", "the smoke", "the coffee"]),
    "build": ("built", ["a house", "a wall", "a bridge", "a sandcastle"]),
    "make": ("made", ["a cake", "a mistake", "a plan", "some tea"]),
    "paint": ("painted", ["the wall", "a picture", "the fence", "the door"]),
    "draw": ("drew", ["a map", "a picture", "a circle", "a horse"]),
    "repair": ("repaired", ["the roof", "the bicycle", "the clock", "the fence"]),
    "plant": ("planted", ["a tree", "the seeds", "some flowers", "potatoes"]),
    "sleep": ("slept", ["all night", "in the tent", "on the sofa", "for hours"]),
    "sit": ("sat", ["on the bench", "by the fire", "at the table", "under the tree"]),
    "stand": ("stood", ["by the door", "in the rain", "on the bridge", "near the window"]),
    "play": ("played", ["football", "the piano", "chess", "in the garden"]),
    "work": ("worked", ["all day", "in the garden", "at the office", "very hard"]),
    "study": ("studied", ["history", "for the exam", "mathematics", "at the library"]),
    "teach": ("taught", ["the children", "history", "music", "the class"]),
    "learn": ("learned", ["to swim", "the song", "a new language", "the rules"]),
    "buy": ("bought", ["a new coat", "some apples", "a ticket", "a house"]),
    "sell": ("sold", ["the old car", "fresh fish", "flowers", "the house"]),
    "open": ("opened", ["the door", "the window", "the box", "the letter"]),
    "close": ("closed", ["the door", "the window", "the shop", "the book"]),
    "carry": ("carried", ["the bags", "the baby", "a heavy box", "the basket"]),
    "throw": ("threw", ["the ball", "a stone", "the rubbish away", "the dice"]),
    "catch": ("caught", ["the ball", "a fish", "the train", "a cold"]),
    "find": ("found", ["the key", "a coin", "the way home", "a lost dog"]),
    "lose": ("lost", ["the key", "the game", "the way", "a glove"]),
    "win": ("won", ["the game", "the race", "a prize", "the match"]),
    "forget": ("forgot", ["the name", "the key", "the address", "the answer"]),
    "remember": ("remembered", ["the name", "the song", "the answer", "the address"]),
    "laugh": ("laughed", ["at the joke", "loudly", "with friends", "all evening"]),
    "cry": ("cried", ["all night", "quietly", "at the film", "with joy"]),
    "smile": ("smiled", ["at the baby", "happily", "at the camera", "warmly"]),
    "think": ("thought", ["about the problem", "carefully", "about home", "for a while"]),
    "help": ("helped", ["the old man", "a friend", "with the dishes", "the neighbours"]),
    "visit": ("visited", ["the museum", "a friend", "the castle", "the zoo"]),
    "wash": ("washed", ["the dishes", "the car", "the clothes", "the windows"]),
    "clean": ("cleaned", ["the kitchen", "the room", "the floor", "the windows"]),
    "go": ("went", ["to the market", "home", "to bed", "to the beach"]),
    "come": ("came", ["home", "to the party", "back", "to the village"]),
    "bring": ("brought", ["a present", "the food", "some flowers", "the book"]),
    "give": ("gave", ["a present", "the money", "a speech", "advice"]),
    "take": ("took", ["the bus", "a photo", "the medicine", "a walk"]),
    "send": ("sent", ["a letter", "a message", "the parcel", "flowers"]),
    "meet": ("met", ["a friend", "the president", "the team", "the new neighbour"]),
    "wear": ("wore", ["a hat", "a red dress", "boots", "a warm coat"]),
    "break": ("broke", ["the window", "a glass", "the record", "a plate"]),
    "fall": ("fell", ["off the ladder", "into the river", "asleep", "down the stairs"]),
    "grow": ("grew", ["tomatoes", "very tall", "roses", "a beard"]),
    "hide": ("hid", ["the key", "behind the door", "under the bed", "the money"]),
    "begin": ("began", ["the lesson", "the journey", "to sing", "the race"]),
    "shop": ("shopped", ["for clothes", "at the market", "for food", "online"]),
    "stop": ("stopped", ["the car", "at the corner", "the music", "talking"]),
    "travel": ("travelled", ["to the coast", "abroad", "by train", "around the world"]),
    "fight": ("fought", ["bravely", "in the war", "the dragon", "for freedom"]),
    "pay": ("paid", ["the bill", "the rent", "for the tickets", "the waiter"]),
    "fish": ("fished", ["in the river", "from the pier", "all morning", "in the lake"]),
}
ING_DOUBLE = {"run", "swim", "sit", "win", "begin", "forget", "shop", "stop", "plan"}
ING_EXCEPT = {"travel": "travelling"}


def ing_form(v):
    if v in ING_EXCEPT:
        return ING_EXCEPT[v]
    if v in ING_DOUBLE:
        return v + v[-1] + "ing"
    if v.endswith("e") and not v.endswith("ee") and v not in {"see"}:
        return v[:-1] + "ing"
    return v + "ing"


def s3_form(v):
    if v in {"go"}:
        return "goes"
    if v.endswith(("s", "sh", "ch", "x")):
        return v + "es"
    if v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


# --- gradable adjectives: base, comparative, superlative, adverb (or None), nouns they describe
ADJ = [
    ("big", "bigger", "biggest", None, ["house", "elephant", "city", "box", "ship", "tree"]),
    ("small", "smaller", "smallest", None, ["house", "mouse", "village", "box", "boat", "room"]),
    ("tall", "taller", "tallest", None, ["tree", "tower", "man", "building", "giraffe", "woman"]),
    ("short", "shorter", "shortest", "shortly", ["story", "film", "road", "man", "speech", "woman"]),
    ("hot", "hotter", "hottest", None, ["tea", "soup", "summer", "day", "coffee", "oven"]),
    ("cold", "colder", "coldest", "coldly", ["water", "winter", "night", "wind", "milk", "day"]),
    ("warm", "warmer", "warmest", "warmly", ["coat", "fire", "day", "bath", "room", "summer"]),
    ("fast", "faster", "fastest", None, ["car", "horse", "train", "runner", "plane", "river"]),
    ("slow", "slower", "slowest", "slowly", ["car", "snail", "train", "turtle", "river", "bus"]),
    ("strong", "stronger", "strongest", "strongly", ["horse", "wind", "man", "rope", "coffee", "bridge"]),
    ("weak", "weaker", "weakest", "weakly", ["horse", "wind", "man", "rope", "tea", "bridge"]),
    ("rich", "richer", "richest", None, ["man", "family", "country", "merchant", "king", "city"]),
    ("poor", "poorer", "poorest", "poorly", ["man", "family", "country", "village", "farmer", "city"]),
    ("happy", "happier", "happiest", "happily", ["child", "family", "dog", "girl", "boy", "day"]),
    ("sad", "sadder", "saddest", "sadly", ["child", "story", "song", "girl", "boy", "day"]),
    ("bright", "brighter", "brightest", "brightly", ["star", "light", "sun", "room", "moon", "colour"]),
    ("dark", "darker", "darkest", None, ["night", "room", "forest", "cave", "sky", "colour"]),
    ("heavy", "heavier", "heaviest", "heavily", ["box", "stone", "bag", "rain", "door", "book"]),
    ("soft", "softer", "softest", "softly", ["pillow", "bed", "voice", "rain", "music", "bread"]),
    ("hard", "harder", "hardest", None, ["stone", "bed", "question", "exam", "floor", "wood"]),
    ("loud", "louder", "loudest", "loudly", ["music", "voice", "noise", "bell", "song", "crowd"]),
    ("quiet", "quieter", "quietest", "quietly", ["room", "voice", "street", "village", "night", "child"]),
    ("clean", "cleaner", "cleanest", None, ["room", "water", "shirt", "kitchen", "floor", "air"]),
    ("dirty", "dirtier", "dirtiest", None, ["room", "water", "shirt", "kitchen", "floor", "air"]),
    ("wide", "wider", "widest", "widely", ["river", "road", "street", "door", "bridge", "valley"]),
    ("narrow", "narrower", "narrowest", "narrowly", ["river", "road", "street", "door", "bridge", "valley"]),
    ("deep", "deeper", "deepest", "deeply", ["lake", "river", "sea", "hole", "well", "cave"]),
    ("high", "higher", "highest", "highly", ["mountain", "wall", "tower", "hill", "price", "cliff"]),
    ("low", "lower", "lowest", None, ["wall", "hill", "price", "table", "voice", "cloud"]),
    ("long", "longer", "longest", None, ["road", "river", "story", "night", "letter", "journey"]),
    ("thick", "thicker", "thickest", "thickly", ["book", "wall", "fog", "coat", "rope", "forest"]),
    ("thin", "thinner", "thinnest", "thinly", ["book", "wall", "paper", "coat", "rope", "ice"]),
    ("young", "younger", "youngest", None, ["man", "woman", "tree", "horse", "child", "king"]),
    ("old", "older", "oldest", None, ["man", "woman", "tree", "castle", "house", "king"]),
    ("brave", "braver", "bravest", "bravely", ["knight", "soldier", "girl", "boy", "firefighter", "dog"]),
    ("calm", "calmer", "calmest", "calmly", ["sea", "lake", "voice", "night", "morning", "horse"]),
    ("wise", "wiser", "wisest", "wisely", ["king", "queen", "teacher", "judge", "owl", "woman"]),
    ("kind", "kinder", "kindest", "kindly", ["woman", "man", "teacher", "nurse", "neighbour", "doctor"]),
    ("sweet", "sweeter", "sweetest", "sweetly", ["cake", "apple", "tea", "song", "voice", "cherry"]),
    ("fresh", "fresher", "freshest", "freshly", ["bread", "fish", "milk", "air", "fruit", "water"]),
    ("wet", "wetter", "wettest", None, ["grass", "road", "dog", "coat", "day", "summer"]),
    ("dry", "drier", "driest", "dryly", ["grass", "road", "bread", "coat", "day", "summer"]),
    ("cheap", "cheaper", "cheapest", "cheaply", ["ticket", "car", "hotel", "coat", "meal", "house"]),
    ("easy", "easier", "easiest", "easily", ["question", "exam", "game", "task", "lesson", "road"]),
    ("safe", "safer", "safest", "safely", ["road", "house", "bridge", "car", "place", "harbour"]),
    ("early", "earlier", "earliest", None, ["train", "morning", "bus", "flight", "spring", "meeting"]),
    ("busy", "busier", "busiest", "busily", ["street", "market", "station", "city", "road", "day"]),
    ("angry", "angrier", "angriest", "angrily", ["man", "dog", "crowd", "king", "woman", "customer"]),
]
OPPOSITES = [("big", "small"), ("tall", "short"), ("hot", "cold"), ("fast", "slow"), ("strong", "weak"),
             ("rich", "poor"), ("happy", "sad"), ("bright", "dark"), ("hard", "soft"), ("loud", "quiet"),
             ("clean", "dirty"), ("wide", "narrow"), ("high", "low"), ("thick", "thin"), ("young", "old"),
             ("wet", "dry")]
ADJ_BY = {a[0]: a for a in ADJ}
# verbs (past tense) that combine with manner adverbs
ADV_VERBS = ["walked", "spoke", "sang", "answered", "smiled", "worked", "drove", "moved", "laughed", "wrote", "danced", "replied"]

# --- animals: (singular, plural, class, sound verb (past), foods, habitat)
ANIMALS = [
    ("dog", "dogs", "pet", "barked", ["bones", "meat"], "house"), ("cat", "cats", "pet", "meowed", ["fish", "milk"], "house"),
    ("hamster", "hamsters", "pet", "squeaked", ["seeds", "nuts"], "house"), ("parrot", "parrots", "pet", "squawked", ["seeds", "fruit"], "house"),
    ("rabbit", "rabbits", "pet", "squeaked", ["carrots", "lettuce"], "garden"),
    ("cow", "cows", "farm", "mooed", ["grass", "hay"], "farm"), ("horse", "horses", "farm", "neighed", ["hay", "oats"], "farm"),
    ("pig", "pigs", "farm", "grunted", ["corn", "scraps"], "farm"), ("goat", "goats", "farm", "bleated", ["grass", "leaves"], "farm"),
    ("chicken", "chickens", "farm", "clucked", ["corn", "seeds"], "farm"), ("duck", "ducks", "farm", "quacked", ["bread", "seeds"], "pond"),
    ("donkey", "donkeys", "farm", "brayed", ["hay", "oats"], "farm"),
    ("lion", "lions", "wild", "roared", ["meat", "zebras"], "savanna"), ("tiger", "tigers", "wild", "roared", ["meat", "deer"], "jungle"),
    ("bear", "bears", "wild", "growled", ["honey", "berries"], "forest"), ("wolf", "wolves", "wild", "howled", ["meat", "deer"], "forest"),
    ("fox", "foxes", "wild", "yelped", ["mice", "chickens"], "forest"), ("elephant", "elephants", "wild", "trumpeted", ["grass", "leaves"], "savanna"),
    ("monkey", "monkeys", "wild", "chattered", ["bananas", "fruit"], "jungle"), ("giraffe", "giraffes", "wild", "hummed", ["leaves", "twigs"], "savanna"),
    ("zebra", "zebras", "wild", "brayed", ["grass", "leaves"], "savanna"), ("deer", "deer", "wild", "bellowed", ["grass", "leaves"], "forest"),
    ("whale", "whales", "sea", "sang", ["krill", "plankton"], "ocean"), ("shark", "sharks", "sea", "circled", ["fish", "seals"], "ocean"),
    ("dolphin", "dolphins", "sea", "whistled", ["fish", "squid"], "ocean"), ("seal", "seals", "sea", "barked", ["fish", "squid"], "ocean"),
    ("octopus", "octopuses", "sea", "hid", ["crabs", "fish"], "ocean"),
    ("eagle", "eagles", "bird", "screeched", ["mice", "fish"], "mountains"), ("owl", "owls", "bird", "hooted", ["mice", "insects"], "woods"),
    ("crow", "crows", "bird", "cawed", ["seeds", "worms"], "woods"), ("sparrow", "sparrows", "bird", "chirped", ["seeds", "crumbs"], "woods"),
    ("swan", "swans", "bird", "hissed", ["weeds", "bread"], "pond"),
    ("bee", "bees", "insect", "buzzed", ["nectar", "pollen"], "meadow"), ("ant", "ants", "insect", "marched", ["crumbs", "sugar"], "meadow"),
    ("butterfly", "butterflies", "insect", "fluttered", ["nectar", "flowers"], "meadow"), ("spider", "spiders", "insect", "waited", ["flies", "insects"], "meadow"),
]

# --- food and drink, colours of objects
FOOD = {
    "fruit": ["apple", "banana", "orange", "grape", "cherry", "lemon", "pear", "peach", "plum", "strawberry", "mango", "melon"],
    "vegetable": ["carrot", "potato", "tomato", "onion", "cabbage", "pea", "bean", "lettuce", "cucumber", "pepper"],
    "bakery": ["bread", "cake", "cookie", "pie", "croissant", "muffin"],
    "dairy": ["milk", "cheese", "butter", "yogurt", "egg", "cream"],
    "meal": ["soup", "salad", "rice", "pasta", "pizza", "sandwich", "stew", "omelette"],
    "drink": ["water", "tea", "coffee", "juice", "wine", "beer", "lemonade", "cocoa"],
}
COLOURS = ["red", "blue", "green", "yellow", "white", "black", "brown", "pink", "purple", "grey", "orange", "gold", "silver"]
COLOUR_OF = {"sky": "blue", "sea": "blue", "grass": "green", "leaf": "green", "snow": "white", "milk": "white",
             "coal": "black", "night": "black", "blood": "red", "cherry": "red", "strawberry": "red", "tomato": "red",
             "banana": "yellow", "lemon": "yellow", "sun": "yellow", "chocolate": "brown", "mud": "brown",
             "pig": "pink", "violet": "purple", "plum": "purple", "cloud": "grey", "elephant": "grey", "carrot": "orange",
             "pumpkin": "orange", "crown": "gold", "ring": "gold", "spoon": "silver", "moon": "silver", "frog": "green",
             "rose": "red", "sunflower": "yellow", "lily": "white", "crow": "black", "swan": "white", "grape": "purple"}
CLOTHES = [("shirt", "shirts"), ("dress", "dresses"), ("coat", "coats"), ("hat", "hats"), ("scarf", "scarves"),
           ("jacket", "jackets"), ("skirt", "skirts"), ("sweater", "sweaters"), ("glove", "gloves"), ("boot", "boots"), ("shoe", "shoes")]
OBJECTS = [("house", "houses"), ("car", "cars"), ("book", "books"), ("table", "tables"), ("chair", "chairs"), ("cup", "cups"),
           ("box", "boxes"), ("key", "keys"), ("knife", "knives"), ("bottle", "bottles"), ("window", "windows"), ("door", "doors"),
           ("tree", "trees"), ("flower", "flowers"), ("bed", "beds"), ("clock", "clocks"), ("lamp", "lamps"), ("bag", "bags"),
           ("church", "churches"), ("bus", "buses"), ("city", "cities"), ("baby", "babies"), ("leaf", "leaves"), ("child", "children"),
           ("foot", "feet"), ("tooth", "teeth"), ("mouse", "mice"), ("glass", "glasses"), ("letter", "letters"), ("river", "rivers")]
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
SEASON_OF = {m: s for s, ms in {"winter": ["december", "january", "february"], "spring": ["march", "april", "may"],
                                  "summer": ["june", "july", "august"], "autumn": ["september", "october", "november"]}.items() for m in ms}
WEATHER = {"winter": ["snowy", "cold", "frosty", "icy"], "spring": ["rainy", "mild", "fresh", "windy"],
           "summer": ["sunny", "hot", "dry", "bright"], "autumn": ["foggy", "windy", "wet", "cloudy"]}
VEHICLES = [("car", "garage", "drove"), ("bus", "station", "rode"), ("train", "station", "rode"), ("plane", "airport", "flew"),
            ("ship", "harbour", "sailed"), ("boat", "harbour", "sailed"), ("bicycle", "shed", "rode"), ("taxi", "rank", "rode"),
            ("tram", "depot", "rode"), ("truck", "depot", "drove")]
PLACES = ["park", "market", "library", "museum", "station", "beach", "village", "town", "church", "school", "hospital",
          "castle", "garden", "forest", "river", "lake", "mountain", "harbour", "bakery", "theatre", "cafe", "zoo"]
EMOTIONS = ["happy", "sad", "angry", "tired", "hungry", "thirsty", "afraid", "bored", "excited", "proud", "lonely", "nervous"]
GIFTS = ["book", "flower", "ring", "watch", "scarf", "letter", "cake", "painting"]


# =============================================================================== generator
class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def c(self, xs):
        return xs[self.r.randrange(len(xs))]

    def person(self, groups=None):
        p = self.c([p for p in PEOPLE if groups is None or p[4] in groups])
        g = self.c("mf")
        return (p[0] if g == "m" else p[1]), g, p

    def fill(self, templates, **kw):
        return self.c(templates).format(**kw)

    # ---- families of sentences ---------------------------------------------------------
    def people(self):
        word, g, p = self.person()
        pr = PRON[g]
        rel_entry = self.c([q for q in PEOPLE if q[4] == "family"])
        rel = self.c([rel_entry[0], rel_entry[1]])
        other, _, _ = self.person()
        kw = dict(p=word, rel=rel, other=other, emo=self.c(EMOTIONS), place=self.c(PLACES), day=self.c(DAYS),
                  gift=self.c(GIFTS), food=self.c(FOOD["meal"] + FOOD["fruit"]), **pr)
        group, age = p[4], p[5]
        general = ["the {p} said that {he} was {emo}", "the {p} told {his} {rel} that {he} felt {emo}",
                   "the {p} went to the {place} on {day}", "the {p} and {his} {rel} went to the {place}",
                   "the {p} looked at {himself} in the mirror", "the {p} gave {his} {rel} a {gift}",
                   "the {p} smiled at the {other}", "the {p} sat down and {he} ate the {food}",
                   "{he} is a {p} and {he} lives near the {place}", "the {p} lost {his} {gift} at the {place}",
                   "nobody knew the {p} but everybody liked {him}", "the {p} washed {himself} and went to bed"]
        specific = {
            "royal": ["the {p} ruled the kingdom from {his} throne", "the {p} wore a golden crown",
                      "the royal {p} lived in the palace", "the {p} spoke to the royal court",
                      "the servants bowed to the {p}", "the {p} and {his} knights rode to the castle",
                      "long live the {p}", "the {p} inherited the throne and the crown",
                      "the {p} held a royal feast in the palace", "the people of the kingdom loved the {p}",
                      "the {p} signed a royal decree", "the {p} reigned over the kingdom for many years"],
            "family": ["the {p} loved {his} children", "the {p} cooked dinner for the whole family",
                       "the {p} lived at home with the family", "the {p} visited {his} parents on {day}",
                       "the {p} came home to {his} family", "the {p} hugged {his} {rel}",
                       "the {p} looked after the children at home", "the family photo showed the {p} and {his} {rel}"],
            "generic": ["a {p} was walking in the park", "the {p} waited at the station",
                        "the {p} asked a question", "the {p} lived in the village", "a {p} knocked at the door",
                        "the {p} crossed the street"],
            "stage": ["the {p} performed on the stage of the theatre", "the {p} starred in a famous film",
                      "the {p} learned {his} lines for the play", "the audience applauded the {p}",
                      "the {p} won an award for the film", "the {p} rehearsed the play"],
            "service": ["the {p} served the food at the restaurant", "the {p} brought the menu to the table",
                        "the {p} poured the wine for the guests", "the customers thanked the {p}",
                        "the {p} cleared the tables in the restaurant", "the {p} took the order"],
            "religious": ["the {p} prayed in the chapel", "the {p} lived a quiet life in the monastery",
                          "the {p} read the holy book", "the {p} rang the bell of the abbey",
                          "the {p} sang hymns in the chapel", "the {p} took a vow of silence"],
        }[group]
        if age == "young":
            specific = specific + ["the young {p} played with {his} toys", "the {p} went to school with {his} friends",
                                   "the little {p} was only {num} years old"]
            kw["num"] = self.c(NUMBERS[2:12])
        if age == "old":
            specific = specific + ["the old {p} told stories by the fire", "the {p} had grey hair and a walking stick",
                                   "the {p} remembered the old days"]
        return self.fill(general if self.r.random() < 0.45 else specific, **kw)

    def plural_people(self):
        p = self.c(PEOPLE)
        g = self.c("mf")
        sg, pl = (p[0], p[2]) if g == "m" else (p[1], p[3])
        num = self.c(NUMBERS[1:12])
        return self.fill(["the {pl} were talking together", "{num} {pl} came to the party", "many {pl} lived in the town",
                          "one {sg} and {num} {pl} sat at the table", "there was a {sg} and there were {num} {pl}",
                          "these {pl} are friends", "this {sg} is a friend", "all the {pl} agreed", "the {pl} are here"],
                         sg=sg, pl=pl, num=num)

    def professions(self):
        prof, places, things, acts = self.c(PROFESSIONS)
        g = self.c("mf")
        kw = dict(prof=prof, place=self.c(places), thing=self.c(things), act=self.c(acts), day=self.c(DAYS),
                  adj=self.c(["kind", "busy", "young", "old", "clever", "tired", "famous", "careful"]), **PRON[g])
        return self.fill(["the {prof} worked at the {place}", "the {prof} used a {thing} at the {place}",
                          "{he} is a {prof} and {he} works at the {place}", "the {prof} {act}",
                          "every day the {prof} goes to the {place}", "the {prof} needed a new {thing}",
                          "the {adj} {prof} {act}", "on {day} the {prof} {act}", "{he} became a {prof} last year",
                          "the {prof} left {his} {thing} at the {place}", "my neighbour is a {prof}",
                          "the {prof} said that {his} {thing} was broken"], **kw)

    def countries(self):
        country, capital, adj, lang, cont = self.c(COUNTRIES)
        o = self.c([c for c in COUNTRIES if c[0] != country])
        word, g, _ = self.person(["generic", "family"])
        kw = dict(country=country, capital=capital, adj=adj, lang=lang, cont=cont, o_country=o[0], o_capital=o[1],
                  p=word, month=self.c(MONTHS), prof=self.c(PROFESSIONS)[0], **PRON[g])
        t = ["{capital} is the capital of {country}", "the capital of {country} is {capital}",
             "the {adj} government met in {capital}", "{country} is a country in {cont}",
             "the {p} flew from {capital} to {o_capital}", "the {p} visited {capital} in {country} in {month}",
             "the {adj} {prof} lived in {capital}", "the train from {capital} to {o_capital} was late",
             "{country} and {o_country} signed a treaty", "the {adj} flag flew over {capital}",
             "the streets of {capital} were busy", "the mayor of {capital} opened the new bridge",
             "the city of {capital} has many museums", "the president of {country} gave a speech",
             "the border of {country} is long", "the {adj} people love their country",
             "{he} was born in {capital} and {he} is {adj}", "the {adj} team played against {o_country}",
             "the airport of {capital} is very busy", "the population of {country} is growing",
             "the {adj} economy of {country} grew last year", "the {p} moved from {country} to {o_country}"]
        if lang:
            t += ["people in {country} speak {lang}", "in {capital} everybody speaks {lang}",
                  "the {p} learned {lang} in {capital}", "{lang} is spoken in {country}"]
        return self.fill(t, **kw)

    def verbs(self):
        base = self.c(list(VERBS))
        past, comps = VERBS[base]
        word, g, _ = self.person()
        kw = dict(base=base, past=past, ing=ing_form(base), s3=s3_form(base), comp=self.c(comps), p=word,
                  day=self.c(DAYS), **PRON[g])
        tense = self.r.randrange(4)
        t = [["{he} will {base} {comp} tomorrow", "the {p} wants to {base} {comp}", "{he} can {base} {comp}",
              "do not {base} {comp}", "they {base} {comp} every day", "let us {base} {comp}",
              "the {p} decided to {base} {comp}", "{he} must {base} {comp} soon"],
             ["yesterday the {p} {past} {comp}", "last {day} {he} {past} {comp}", "{he} {past} {comp} an hour ago",
              "the {p} {past} {comp} last year", "once upon a time a {p} {past} {comp}", "the {p} already {past} {comp}"],
             ["the {p} is {ing} {comp} now", "{he} was {ing} {comp} when the phone rang",
              "they are {ing} {comp} at the moment", "the {p} is still {ing} {comp}",
              "{he} was {ing} {comp} all afternoon", "look the {p} is {ing} {comp}"],
             ["the {p} {s3} {comp} every day", "{he} often {s3} {comp}", "{he} usually {s3} {comp} in the morning",
              "the {p} always {s3} {comp}", "every {day} {he} {s3} {comp}", "{he} never {s3} {comp}"]][tense]
        return self.fill(t, **kw)

    def adjectives(self):
        base, cmp_, sup, adv, nouns = self.c(ADJ)
        n1, n2 = self.c(nouns), self.c(nouns)
        word, g, _ = self.person()
        kind = self.r.randrange(4 if adv else 3)
        kw = dict(a=base, cmp=cmp_, sup=sup, adv=adv, n=n1, n2=n2, place=self.c(PLACES), p=word, v=self.c(ADV_VERBS), **PRON[g])
        t = [["the {n} was very {a}", "a {a} {n} stood near the {place}", "{he} saw a {a} {n}", "the {n} looked {a}",
              "it was a {a} {n}", "what a {a} {n}", "the {n} is quite {a}", "the {n} was too {a}"],
             ["the {n} is {cmp} than the {n2}", "this {n} is much {cmp} than that one", "the {n} became {cmp} and {cmp}",
              "{he} wanted a {cmp} {n}", "nothing is {cmp} than this {n}", "the new {n} was {cmp} than the old one"],
             ["the {n} is the {sup} {n} in the {place}", "it was the {sup} {n} {he} had ever seen", "the {sup} {n} of all",
              "this is the {sup} {n} in the world", "the {sup} one was the {n}", "{he} chose the {sup} {n}"],
             ["the {p} {v} {adv}", "{he} {v} very {adv}", "{he} {v} {adv} and then left", "the {p} {v} {adv} all day"]][kind]
        return self.fill(t, **kw)

    def animals(self):
        sg, pl, cls, sound, foods, home = self.c(ANIMALS)
        other = self.c([a for a in ANIMALS if a[2] == cls])
        word, g, _ = self.person()
        kw = dict(a=sg, ap=pl, sound=sound, food=self.c(foods), home=home, b=other[0], bp=other[1], cls=cls,
                  num=self.c(NUMBERS[1:12]), p=word, **PRON[g])
        t = ["the {a} {sound}", "the {a} {sound} at the {p}", "the {a} ate the {food}", "the {a} lives in the {home}",
             "{he} saw a {a} in the {home}", "the {a} and the {b} were friends", "{num} {ap} were in the {home}",
             "many {ap} live in the {home}", "the {ap} {sound} all night", "the {ap} like {food}",
             "a {a} is an animal", "one {a} and {num} {ap}", "the {ap} and the {bp} ate the {food}",
             "the {p} fed the {a} some {food}"]
        if cls == "pet":
            t += ["the {p} took {his} {a} to the vet", "{his} pet {a} slept on the sofa"]
        if cls == "farm":
            t += ["the farmer fed the {ap} on the farm", "the {a} stood in the barn"]
        if cls == "wild":
            t += ["the wild {a} hunted in the {home}", "we saw a {a} at the zoo"]
        if cls == "sea":
            t += ["the {a} swam deep in the ocean", "the {a} dived under the waves"]
        if cls == "bird":
            t += ["the {a} flew over the trees", "the {a} built a nest"]
        if cls == "insect":
            t += ["the tiny {a} crawled on the leaf", "the {a} visited the flowers"]
        return self.fill(t, **kw)

    def food(self):
        cat = self.c(list(FOOD))
        f, f2 = self.c(FOOD[cat]), self.c(FOOD[cat])
        word, g, _ = self.person()
        kw = dict(f=f, f2=f2, p=word, meal=self.c(["breakfast", "lunch", "dinner"]), **PRON[g])
        t = {"fruit": ["the {p} ate a {f} for {meal}", "a ripe {f} fell from the tree", "fresh {f} and {f2} from the market",
                       "the {f} was sweet and juicy", "{he} peeled the {f}", "a bowl of {f} and {f2}"],
             "vegetable": ["the {p} grew {f} in the garden", "the soup was made with {f} and {f2}",
                           "{he} chopped the {f} for the salad", "the farmer sold {f} at the market", "boiled {f} with butter"],
             "bakery": ["the baker sold the {f}", "a slice of {f} with tea", "the {f} came out of the oven",
                        "{he} baked a {f} for the party", "the {f} smelled delicious"],
             "dairy": ["the {p} bought {f} and {f2} at the shop", "{he} put the {f} in the fridge",
                       "the {f} came from the farm", "the {f} was fresh"],
             "meal": ["the {p} cooked {f} for {meal}", "{he} ordered the {f} at the restaurant",
                      "we had {f} and {f2} for {meal}", "the {f} was hot and tasty"],
             "drink": ["the {p} drank a cup of {f}", "{he} poured a glass of {f}", "a hot {f} on a cold day",
                       "the {p} ordered a {f} at the cafe", "{he} prefers {f} to {f2}", "a bottle of {f}"]}[cat]
        return self.fill(t, **kw)

    def colours(self):
        r = self.r.random()
        word, g, _ = self.person()
        if r < 0.6:
            obj = self.c(list(COLOUR_OF))
            return self.fill(["the {o} was {c}", "a {c} {o}", "the {o} is always {c}", "{c} like the {o}",
                              "the colour of the {o} is {c}"], o=obj, c=COLOUR_OF[obj])
        cl = self.c(CLOTHES)[0]
        return self.fill(["the {p} wore a {c} {cl}", "{he} bought a {c} {cl} at the shop", "the {p} painted the door {c}",
                          "{his} favourite colour is {c}", "a {c} car stopped outside"],
                         p=word, c=self.c(COLOURS), cl=cl, **PRON[g])

    def plurals(self):
        sg, pl = self.c(OBJECTS + CLOTHES)
        num = self.c(NUMBERS[1:12])
        return self.fill(["one {sg} and {num} {pl}", "there was a {sg} and there were {num} {pl}", "{num} {pl}",
                          "these {pl} are new", "this {sg} is new", "many {pl}", "a {sg}", "the {pl} were old",
                          "the {sg} was old", "how many {pl} are there", "every {sg} is different",
                          "all the {pl} are different"], sg=sg, pl=pl, num=num)

    def numbers(self):
        a, b = self.r.randrange(1, 11), self.r.randrange(1, 11)
        k = self.r.randrange(5)
        if k == 0:
            return f"{NUMBERS[a - 1]} plus {NUMBERS[b - 1]} equals {NUMBERS[a + b - 1]}"
        if k == 1:
            n = self.r.randrange(1, 20)
            return f"{NUMBERS[n]} comes after {NUMBERS[n - 1]}"
        if k == 2:
            i = self.r.randrange(9)
            return f"the {ORDINALS[i + 1]} comes after the {ORDINALS[i]}"
        if k == 3:
            i = self.r.randrange(10)
            return self.c([f"{NUMBERS[i]} is the number before {NUMBERS[i + 1]}", f"the {ORDINALS[i]} day of the trip",
                           f"{ORDINALS[i]} prize went to number {NUMBERS[i]}"])
        n = self.r.randrange(2, 20)
        sg, pl = self.c(OBJECTS)
        return f"{NUMBERS[n - 1]} {pl} cost {NUMBERS[n]} pounds"

    def time(self):
        k = self.r.randrange(4)
        word, g, _ = self.person()
        if k == 0:
            i = self.r.randrange(7)
            return self.fill(["{d} comes after {pd}", "on {d} morning {he} went to the {place}", "the shop is closed on {d}",
                              "see you on {d}", "the meeting is on {d} and {nd}"],
                             d=DAYS[i], pd=DAYS[i - 1], nd=DAYS[(i + 1) % 7], place=self.c(PLACES), **PRON[g])
        if k == 1:
            i = self.r.randrange(12)
            m = MONTHS[i]
            s = SEASON_OF[m]
            return self.fill(["{m} comes after {pm}", "in {m} the weather was {w}", "{m} is a {s} month",
                              "{he} was born in {m}", "the holiday starts in {m}", "{m} and {nm} are in {s}"],
                             m=m, pm=MONTHS[i - 1], nm=MONTHS[(i + 1) % 12], s=s, w=self.c(WEATHER[s]), **PRON[g])
        s = self.c(list(WEATHER))
        return self.fill(["in {s} the days are {w}", "{s} is {w} and {w2}", "the {s} was {w} this year",
                          "{he} likes {s} because it is {w}", "a {w} {s} morning"],
                         s=s, w=self.c(WEATHER[s]), w2=self.c(WEATHER[s]), **PRON[g])

    def transport(self):
        veh, home, verb = self.c(VEHICLES)
        country = self.c(COUNTRIES)
        word, g, _ = self.person()
        return self.fill(["the {p} went to {cap} by {v}", "the {v} was waiting at the {home}", "{he} {verb} the {v} to {cap}",
                          "the {v} left the {home} on time", "the {v} was {fs}", "{he} missed the {v}",
                          "the {v} arrived at the {home}"],
                         p=word, v=veh, home=home, verb=verb, cap=country[1], fs=self.c(["fast", "slow", "late", "full"]), **PRON[g])

    def sentence(self):
        fams = [(self.people, 16), (self.plural_people, 4), (self.professions, 12), (self.countries, 14),
                (self.verbs, 16), (self.adjectives, 14), (self.animals, 10), (self.food, 7), (self.colours, 5),
                (self.plurals, 5), (self.numbers, 3), (self.time, 5), (self.transport, 4)]
        tot = sum(w for _, w in fams)
        x = self.r.random() * tot
        for f, w in fams:
            x -= w
            if x < 0:
                return f()
        return fams[-1][0]()


# ---- ground-truth categories (for evaluation + simulator highlighting only) -------------
def categories():
    cat = {}

    def put(words, name):
        for w in words:
            cat.setdefault(w, name)

    put([p[i] for p in PEOPLE for i in (0, 1) if p[4] == "royal"], "royalty")
    put([p[i] for p in PEOPLE for i in (0, 1) if p[4] == "family"], "family")
    put([p[i] for p in PEOPLE for i in (0, 1)], "people")
    put([p[i] for p in PEOPLE for i in (2, 3)], "people (plural)")
    put([c[0] for c in COUNTRIES], "countries")
    put([c[1] for c in COUNTRIES], "capitals")
    put([c[2] for c in COUNTRIES] + [c[3] for c in COUNTRIES if c[3]], "nationalities / languages")
    put([p[0] for p in PROFESSIONS], "professions")
    put(list(VERBS), "verbs (base)")
    put([v[0] for v in VERBS.values()], "verbs (past)")
    put([ing_form(v) for v in VERBS], "verbs (-ing)")
    put([s3_form(v) for v in VERBS], "verbs (3rd person)")
    put([a[0] for a in ADJ], "adjectives")
    put([a[1] for a in ADJ], "comparatives")
    put([a[2] for a in ADJ], "superlatives")
    put([a[3] for a in ADJ if a[3]], "adverbs")
    put([a[0] for a in ANIMALS] + [a[1] for a in ANIMALS], "animals")
    for k, v in FOOD.items():
        put(v, "drinks" if k == "drink" else "food")
    put(COLOURS, "colours")
    put([c[0] for c in CLOTHES] + [c[1] for c in CLOTHES], "clothes")
    put(NUMBERS + ORDINALS, "numbers")
    put(DAYS, "days")
    put(MONTHS, "months")
    put(list(WEATHER), "seasons")
    put([o[0] for o in OBJECTS] + [o[1] for o in OBJECTS], "objects")
    put([v[0] for v in VEHICLES], "vehicles")
    put(PLACES, "places")
    put(EMOTIONS, "emotions")
    return cat


def analogy_sets():
    """Ground-truth (a, b) pairs per relation; questions are a:b :: c:d for all ordered pairs of pairs."""
    S = {
        "male - female": [(p[0], p[1]) for p in PEOPLE],
        "capital - country": [(c[1], c[0]) for c in COUNTRIES],
        "country - nationality": [(c[0], c[2]) for c in COUNTRIES],
        "verb - past tense": [(v, VERBS[v][0]) for v in VERBS],
        "verb - present participle": [(v, ing_form(v)) for v in VERBS],
        "verb - 3rd person": [(v, s3_form(v)) for v in VERBS],
        "adjective - comparative": [(a[0], a[1]) for a in ADJ],
        "adjective - superlative": [(a[0], a[2]) for a in ADJ],
        "adjective - adverb": [(a[0], a[3]) for a in ADJ if a[3]],
        "singular - plural": [(o[0], o[1]) for o in OBJECTS + CLOTHES] + [(a[0], a[1]) for a in ANIMALS if a[0] != a[1]],
        "opposites": OPPOSITES,
    }
    return S


# =============================================================================== training
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    gen = Gen(SEED)
    sents = [gen.sentence().split() for _ in range(N_SENT)]
    text = "\n".join(" ".join(s) for s in sents) + "\n"
    if "--dump" in sys.argv:
        Path("toyland.txt").write_text(text, encoding="utf-8")
    n_tokens = sum(len(s) for s in sents)
    counts = Counter(w for s in sents for w in s)
    vocab = [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= MIN_COUNT]
    stoi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    freq = np.array([counts[w] for w in vocab], dtype=np.float64)
    print(f"corpus: {len(sents):,} sentences, {n_tokens:,} tokens, {len(text.encode()):,} bytes, "
          f"{len(counts):,} word types, vocabulary (count>={MIN_COUNT}) {V:,}")

    rng = np.random.default_rng(SEED)
    f = freq / freq.sum()
    keep_p = np.minimum(1.0, np.sqrt(SUBSAMPLE_T / f) + SUBSAMPLE_T / f)
    noise = freq ** 0.75
    noise /= noise.sum()
    noise_cdf = np.cumsum(noise)
    ids_sents = [np.array([stoi[w] for w in s if w in stoi], dtype=np.int64) for s in sents]

    flat = np.concatenate(ids_sents)
    sent_id = np.concatenate([np.full(len(s), k) for k, s in enumerate(ids_sents)])

    def make_pairs():
        keep = rng.random(len(flat)) < keep_p[flat]                   # subsampling, fresh every epoch
        toks, sid = flat[keep], sent_id[keep]
        b = rng.integers(1, WINDOW + 1, size=len(toks))               # dynamic window per centre word
        cs, os_ = [], []
        for off in range(1, WINDOW + 1):                              # context at distance `off` on each side
            same = sid[off:] == sid[:-off]                            # never cross a sentence boundary
            right = same & (b[:-off] >= off)                          # centre i, context i + off
            cs.append(toks[:-off][right]); os_.append(toks[off:][right])
            left = same & (b[off:] >= off)                            # centre i + off, context i
            cs.append(toks[off:][left]); os_.append(toks[:-off][left])
        return np.concatenate(cs), np.concatenate(os_)

    emb_in = nn.Embedding(V, DIM, sparse=True)
    emb_out = nn.Embedding(V, DIM, sparse=True)
    with torch.no_grad():
        emb_in.weight.uniform_(-0.5 / DIM, 0.5 / DIM)                # word2vec's initialisation
        emb_out.weight.zero_()
    opt = torch.optim.SparseAdam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=LR)

    epoch_pairs = [make_pairs() for _ in range(EPOCHS)]
    total_steps = sum(math.ceil(len(c) / BATCH) for c, _ in epoch_pairs)
    step = 0
    for ep in range(EPOCHS):
        c_all, o_all = epoch_pairs[ep]
        perm = rng.permutation(len(c_all))
        c_all, o_all = c_all[perm], o_all[perm]
        negs_all = np.searchsorted(noise_cdf, rng.random((len(c_all), NEG)))
        negs_all = np.minimum(negs_all, V - 1)
        tot, nb = 0.0, 0
        for k in range(0, len(c_all), BATCH):
            lr = LR * max(1e-4, 1 - step / total_steps)
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            c = torch.from_numpy(c_all[k:k + BATCH])
            o = torch.from_numpy(o_all[k:k + BATCH])
            ng = torch.from_numpy(negs_all[k:k + BATCH])
            v = emb_in(c)                                             # (B, d)
            pos = (emb_out(o) * v).sum(-1)                            # (B,)
            neg = torch.bmm(emb_out(ng), v.unsqueeze(-1)).squeeze(-1)  # (B, K)
            loss = -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1; step += 1
        print(f"epoch {ep + 1}/{EPOCHS}: {len(c_all):,} pairs, mean SGNS loss {tot / nb:.4f}  ({time.time() - t0:.0f}s)")

    W = emb_in.weight.detach().numpy().astype(np.float64)
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    # ---- quantise (per-word scale) and continue with EXACTLY the de-quantised vectors
    scale = np.abs(W).max(axis=1) / QMAX
    Q = np.round(W / scale[:, None]).astype(np.int64)
    Wq = Q * scale[:, None]
    Wn = Wq / np.linalg.norm(Wq, axis=1, keepdims=True)
    print(f"quantisation: max |cos(original, quantised)| error {np.max(1 - np.sum(W * Wn, axis=1)):.2e}")

    def neighbours(w, k=10):
        sims = Wn @ Wn[stoi[w]]
        sims[stoi[w]] = -np.inf
        idx = np.argsort(-sims, kind="stable")[:k]
        return [(vocab[i], float(sims[i])) for i in idx]

    def analogy(a, b, c, k=5):
        q = Wn[stoi[b]] - Wn[stoi[a]] + Wn[stoi[c]]
        sims = Wn @ q
        for w in (a, b, c):
            sims[stoi[w]] = -np.inf
        idx = np.argsort(-sims, kind="stable")[:k]
        return [(vocab[i], float(sims[i] / np.linalg.norm(q))) for i in idx]

    # ---- analogy evaluation (3CosAdd, inputs excluded) and the "only c" baseline
    evals = []
    tot_q = tot_hit = tot_base = 0
    for name, pairs in analogy_sets().items():
        pairs = [(a, b) for a, b in pairs if a in stoi and b in stoi]
        A = np.array([stoi[a] for a, _ in pairs]); B = np.array([stoi[b] for _, b in pairs])
        n_q = hit = base = 0
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i == j:
                    continue
                a, b, c, d = A[i], B[i], A[j], B[j]
                q = Wn[b] - Wn[a] + Wn[c]
                s = Wn @ q
                s[[a, b, c]] = -np.inf
                hit += int(np.argmax(s) == d)
                s2 = Wn @ Wn[c]
                s2[[a, b, c]] = -np.inf
                base += int(np.argmax(s2) == d)
                n_q += 1
        evals.append({"relation": name, "questions": n_q, "top1": round(hit / n_q, 4), "baseline": round(base / n_q, 4)})
        tot_q += n_q; tot_hit += hit; tot_base += base
        print(f"  analogy {name:28s} {n_q:5d} questions  3CosAdd top-1 {hit / n_q:6.1%}   'nearest to c' baseline {base / n_q:6.1%}")
    print(f"  analogy OVERALL {tot_q} questions: top-1 {tot_hit / tot_q:.1%}  (baseline {tot_base / tot_q:.1%})")

    # ---- neighbour purity: fraction of the 10 nearest neighbours sharing the word's category
    cats = categories()
    cat_names = sorted(set(cats.values()))
    labelled = [w for w in vocab if w in cats]
    pur = []
    for w in labelled:
        nb = neighbours(w, 10)
        pur.append(np.mean([cats.get(x) == cats[w] for x, _ in nb]))
    print(f"  neighbour purity (top-10 share the ground-truth category): {np.mean(pur):.1%} over {len(labelled)} words")
    rand_pur = np.mean([sum(1 for x in labelled if cats[x] == cats[w]) / V for w in labelled])
    print(f"  (chance level for a random neighbour: {rand_pur:.1%})")

    for w in ["king", "paris", "swim", "happy", "dog", "january", "doctor", "seven", "orange"]:
        print(f"  {w:8s} -> " + ", ".join(f"{x} {s:.2f}" for x, s in neighbours(w, 6)))
    examples = [("man", "king", "woman"), ("france", "paris", "italy"), ("walk", "walked", "swim"),
                ("big", "bigger", "small"), ("germany", "german", "spain"), ("dog", "dogs", "mouse"),
                ("prince", "princess", "son"), ("slow", "slowly", "quiet")]
    for a, b, c in examples:
        print(f"  {b} - {a} + {c} = " + ", ".join(f"{x} {s:.2f}" for x, s in analogy(a, b, c, 3)))

    # ---- bias probe: projection of profession words on the he-she direction
    gdir = Wn[stoi["he"]] - Wn[stoi["she"]]
    gdir /= np.linalg.norm(gdir)
    proj = sorted(((float(Wn[stoi[p[0]]] @ gdir), p[0]) for p in PROFESSIONS if p[0] in stoi))
    print("  he-she projection of professions: min %.3f (%s), max %.3f (%s); for comparison king %.3f, queen %.3f" % (
        proj[0][0], proj[0][1], proj[-1][0], proj[-1][1], Wn[stoi['king']] @ gdir, Wn[stoi['queen']] @ gdir))

    # ---- 2-D projections
    pca = PCA(n_components=2, random_state=0).fit(Wn)
    P = pca.transform(Wn)
    print(f"  PCA: first two components explain {pca.explained_variance_ratio_.sum():.1%} of the variance")
    T = TSNE(n_components=2, perplexity=30, init="pca", random_state=0, metric="cosine").fit_transform(Wn)

    def norm2(X):
        X = X - X.mean(0)
        return X / np.abs(X).max()

    P, T = norm2(P), norm2(T)

    verify = {
        "neighbours": [{"word": w, "top": [[x, round(s, 6)] for x, s in neighbours(w, 10)]} for w in ["king", "paris", "swim", "seven"]],
        "analogies": [{"abc": [a, b, c], "top": [[x, round(s, 6)] for x, s in analogy(a, b, c, 5)]} for a, b, c in examples[:4]],
    }
    out = {
        "format": "word2vec-sgns-v1",
        "description": "Skip-gram with negative sampling trained by web/scripts/train-word2vec.py on a generated 'Toyland English' corpus.",
        "corpus": {"sentences": len(sents), "tokens": int(n_tokens), "bytes": len(text.encode()), "types": len(counts),
                   "kind": "generated from a hand-written lexicon and sentence templates (no external text)"},
        "config": {"dim": DIM, "window": WINDOW, "negatives": NEG, "subsample_t": SUBSAMPLE_T, "epochs": EPOCHS,
                   "min_count": MIN_COUNT, "optimizer": f"SparseAdam lr {LR} linear decay", "batch": BATCH},
        "words": vocab,
        "counts": [int(counts[w]) for w in vocab],
        "catNames": cat_names,
        "cat": [cat_names.index(cats[w]) if w in cats else -1 for w in vocab],
        "scale": [float(f"{s:.6g}") for s in scale],
        "q": Q.reshape(-1).tolist(),
        "pca": [[round(float(x), 3), round(float(y), 3)] for x, y in P],
        "tsne": [[round(float(x), 3), round(float(y), 3)] for x, y in T],
        "analogyEval": evals,
        "analogyOverall": {"questions": tot_q, "top1": round(tot_hit / tot_q, 4), "baseline": round(tot_base / tot_q, 4)},
        "purity": round(float(np.mean(pur)), 4),
        "verify": verify,
    }
    # The stored scale is rounded to 6 significant digits: recompute the verification from it.
    scale_r = np.array(out["scale"])
    Wq = Q * scale_r[:, None]
    Wn[:] = Wq / np.linalg.norm(Wq, axis=1, keepdims=True)
    out["verify"] = {
        "neighbours": [{"word": w, "top": [[x, round(s, 6)] for x, s in neighbours(w, 10)]} for w in ["king", "paris", "swim", "seven"]],
        "analogies": [{"abc": [a, b, c], "top": [[x, round(s, 6)] for x, s in analogy(a, b, c, 5)]} for a, b, c in examples[:4]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
