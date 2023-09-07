# initial categories generated using chatgpt (chat.openai.com) with the following prompt:



import dsp
import json
from random import sample, choice, random
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

english_stopwords = stopwords.words('english')

with open("auth.json") as f:
    secrets = json.load(f)

OPENAI_KEY = secrets["OPENAI_KEY"]

#number of categories to generate
N_CATEGORIES = 600

lm = dsp.GPT3(model='gpt-3.5-turbo', api_key=OPENAI_KEY, model_type='chat')
dsp.settings.configure(lm=lm)


#adjectives, nouns, verbs, and a couple tiny categories
descriptive_tags = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "IN", "FW", "RBS", "VB", "VBG", "VBN", "VBD", "VBN", "VBP", "VBZ"]

def valid_word(word, tag=None, good_tags=descriptive_tags):
    if tag is None:
        word, tag = nltk.pos_tag([word])[0]
    return word not in english_stopwords and len(word) > 2 and tag in good_tags


def main(gold_standard_categories_filename: str, prev_categories_filename: str, wordlist_filename: str, outfile=f"cache/{N_CATEGORIES}_categories_gpt.list.txt"):
    """Generate categories using few-shot learning

    Args:
        gold_standard_categories_filename (str): base set of example categories (can be from exisitng research ex. SEMCAT or human generated)
        prev_categories_filename (str): if the script has been run before, append to new results this file
        wordlist_filename (str): we will prompt gpt to generate categories the match words in this list
        outfile (_type_, optional): where to store the generated list. Defaults to f"cache/{N_CATEGORIES}_categories_gpt.list.txt".
    """
    
    train = []
    categories = {}
    with open(gold_standard_categories_filename) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            category = line.split(":")[0].strip()
            words = line.split(":")[1].strip()
            train.append((f"What are nouns, concepts, or words that can be associated with {category}?", [words]))
    
    train = [dsp.Example(question=question, answer=answer) for question, answer in train]
    
    with open(prev_categories_filename) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            category = line.split(":")[0].strip()
            words = line.split(":")[1].strip()
            categories[category] = words

    with open(wordlist_filename) as f:
        wordlist = [word.strip().lower() for word in f.readlines()]

    Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
    Answer = dsp.Type(prefix="Answer:", desc="${a comma-seperated list of about 10-20 english words}", format=dsp.format_answers)
    categories_template = dsp.Template(instructions=f"You are teaching an English as a Second Language (ESL) class. To help your students learn new vocab, you have decided to divide the words into groups. In this task, you will be asked to think of words that should be part of the given group.", question=Question(), answer=Answer())
    
    try:
        with tqdm(total = N_CATEGORIES - len(categories.keys())) as progress_bar:
            while(len(categories.keys()) < N_CATEGORIES):
                if len(wordlist) < 1:
                    print("out of words")
                    break

                word = choice(wordlist)

                if word in categories or not valid_word(word):
                    wordlist.remove(word)
                    continue

                q = f"What are nouns, concepts, or words that can be associated with {word}?"
                demos = dsp.sample(train, k=8)

                example = dsp.Example(question=q, demos=demos)
                _, completions = dsp.generate(categories_template)(example, stage='qa')
                categories[word] = completions.answer
                progress_bar.update(1)

                if random() < .2:
                    lm.inspect_history()
    finally:
        with open(outfile, "w+") as f:
            for c,wl in categories.items():
                print(f"{c}: {wl}", file=f)


if __name__ == '__main__':
    #main("cache/gold_standard_categories.list.txt", "cache/250_categories_gpt.list.txt", "data/wordlist_mostcommon1k.txt")
    pass