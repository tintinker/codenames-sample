from collections import defaultdict
from typing import List
import dsp
from random import choice, random
from tqdm import tqdm
from torchtext.vocab import GloVe
import numpy as np
import json
import yaml

with open("auth.json") as f:
    secrets = json.load(f)

OPENAI_KEY = secrets["OPENAI_KEY"]


lm = dsp.GPT3(model='gpt-3.5-turbo', api_key=OPENAI_KEY, model_type='chat')
dsp.settings.configure(lm=lm)

def read_examples_from_file(file_path: str):
    """# Read the audit (manually human annotated examples) from the yaml file
    Args:
        file_path (str): yaml file

    Returns:
        _type_: Tuple[positive examples, negative examples]
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    pos = data['positive_examples']
    neg = data['negative_examples']

    positive_examples = []
    negative_examples = []

    for item in pos:
        word, options, correct_choices = item
        positive_examples += [(word, options, correct_choices)]
        
    for item in neg:
        word, options, correct_choices = item
        negative_examples += [(word, options, correct_choices)]
    
    return positive_examples, negative_examples

def cosine_similarity(A, B):
    dot_product = np.dot(A, B.T)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B, axis=1)
    similarities = dot_product / (norm_A * norm_B)
    return similarities.reshape(-1)

def find_closest_words(query_word: str, corpus_words: List[str], glove, top_n=10) -> List[str]:
    """Find top_n closest words from the corpus by glove embedding cosine similarity to the query word, 

    Args:
        query_word (str): 
        corpus_words (List[str]): 
        glove (torchtext.vocab.Glove): 
        top_n (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: List[str]
    """
    
    # Check if query word is in GloVe vocabulary
    if query_word not in glove.stoi:
        print(f"'{query_word}' is not in the GloVe vocabulary.")
        return
    
    # Get the word vectors for the query word and corpus words
    query_vector = glove[query_word]
    corpus_words = [word for word in corpus_words if word in glove.stoi]
    corpus_vectors = np.array([glove[word] for word in corpus_words])
    
    # Calculate cosine similarities between the query vector and corpus vectors
    similarity_scores = cosine_similarity(query_vector, corpus_vectors)
    
    idxs = np.argsort(similarity_scores)[::-1][:top_n]
    return [corpus_words[i] for i in idxs]

def audit(category_file, audit_training_examples_file, tests_outfile="audit_tests.txt", results_outfile="audit_results.txt"):
    """For each category, we'll use the training examples and a basic cosine similarity to probe GPT-3: should any of the top ten nearest words be included in the given category?

    Args:
        category_file (str): filename for the output of 'gpt_categories.py'
        audit_training_examples_file (str): human annotated examples
        tests_outfile (str, optional): _description_. Defaults to "audit_tests.txt".
        results_outfile (str, optional): _description_. Defaults to "audit_results.txt".
    """
    
    Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
    Answer = dsp.Type(prefix="Answer:", desc="${A yes or no answer, and if relevant, the words that should be added to the group.}", format=dsp.format_answers)
    audit_template = dsp.Template(instructions=f"You are teaching an English as a Second Language (ESL) class. To help your students learn new vocab, you have decided to divide the words into groups. Before you finalize your groups, you want to make sure they are both comprehensive and representative. In this task, you will be asked whether any of the provided words should be added to the given group.", question=Question(), answer=Answer())
   
    
    positive_examples, negative_examples = read_examples_from_file(audit_training_examples_file)
    
    positive_train = [
        (f"Should any of {', '.join(options)} be included in the group: {category}?", [f"Yes: {', '.join(selected)}"]) for category, options, selected in positive_examples
    ]

    negative_train = [
        (f"Should any of {', '.join(options)} be included in the group: {category}?", ["No."]) for category, options, _ in negative_examples
    ]


    positive_train = [dsp.Example(question=question, answer=answer) for question, answer in positive_train]
    negative_train = [dsp.Example(question=question, answer=answer) for question, answer in negative_train]

    demos = dsp.sample(positive_train, k=5) + negative_train

    glove = GloVe(name='6B', dim=300)
    vocab = set()
    categories = defaultdict(set)
    with open(category_file) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            category = line.split(":")[0].strip()
            words = [w.split() for w in line.split(":")[1].strip().split(", ")]
            flattened_words = [item for sublist in words for item in sublist]
            vocab.update(flattened_words)
            categories[category].update(flattened_words)

    print("num vocab", len(vocab))

    with open(tests_outfile, "w+") as f:
        for category in tqdm(categories.keys()):
                tqdm.write(f"Auditing: {category}\t")
                closest_words = find_closest_words(category, list(vocab - categories[category]), glove, top_n=20)
                if closest_words:
                    q = f"Should any of {', '.join(closest_words)} be included in the group: {category}?"
                    print(category, q, file=f)

    audit_results = {}
    try:
        for category in tqdm(categories.keys()):
            tqdm.write(f"Auditing: {category}\t")
            closest_words = find_closest_words(category, list(vocab - categories[category]), glove, top_n=20)
            if closest_words:
                q = f"Should any of {', '.join(closest_words)} be included in the group: {category}?"
                example = dsp.Example(question=q, demos=demos)
                
                yes_votes = []
                no_votes = []
                for _ in range(5):
                    _, completions = dsp.generate(audit_template, temperature=0.7)(example, stage='qa')
                    if "Yes" in completions.answer:
                        yes_votes.append(completions.answer)
                    else:
                        no_votes.append(completions.answer)
                if len(yes_votes) > len(no_votes):
                    audit_results[category] = choice(yes_votes)
                    tqdm.write(f"Decided Yes: ({len(yes_votes)}/5)")
                else:
                    audit_results[category] = choice(no_votes)
                    tqdm.write(f"Decided No: ({len(no_votes)}/5)")


            if random() < .2:
                lm.inspect_history(2)
    finally:
        with open(results_outfile, "w+") as f:
            for c,r in audit_results.items():
                print(f"{c}: {r}", file=f)

def parse_audit_result(file_path):
    """Parse GPT3's answers

    Args:
        file_path (str): output of main audit function. Contains GPT3's answers

    Returns:
        _type_: _description_
    """
    result = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(': ')
        topic = parts[0]
        words = []

        if len(parts) > 1 and parts[1].startswith('Yes'):
            words = parts[1][5:].strip().split(', ')

        result.append((topic, words))

    return result

def incorporate_audit_results(categories_filename, audit_results_filename, outfile="categories_with_audit.txt"):
    categories = defaultdict(set)

    with open(categories_filename) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            category = line.split(":")[0].strip()
            words = [w.split() for w in line.split(":")[1].strip().split(", ")]
            flattened_words = [item for sublist in words for item in sublist]
            categories[category].update(flattened_words)

    print(len(categories))
    for topic, words in parse_audit_result(audit_results_filename):
        categories[topic].update(words)
    print(len(categories))

    with open(outfile, "w+") as f:
        for c,ws in categories.items():
            print(f"{c}: {', '.join([w for w in ws if w]).replace(',,',',')}", file=f)

if __name__ == '__main__':
    #incorporate_audit_results("cache/600_categories_gpt.list.txt", "cache/600_gpt_audit_results.txt", "cache/600_categories_gpt_post_audit.list.txt")
    #audit("cache/600_categories_gpt.list.txt")
    pass