from utils import parse_non_unique_words
import pandas as pd

non_unique_words_fp = "./data/NonUniqueWords_edited.xlsx"
non_unique_words = parse_non_unique_words(non_unique_words_fp)

with open('./data/behavior_triplets_train90.txt', 'r') as file:
    triplets_from_txt = [line.strip().split() for line in file]

concept_fp = "./data/things_concepts.tsv"
concepts = pd.read_csv(concept_fp, sep="\t")

# concepts.iloc[index]["uniqueID"]

def not_unique(words: str) -> bool:
    """checks if the triplet contains non-unique words, e.g. words that are not twice
    in the things object images and therefore might be ambiguous for when collecting
    gpt3 ooo judgments based on words"""
    words = set(words)
    n = len(words.intersection(non_unique_words))
    return True if n > 0 else False

# go through behav triplets and only copy lines that non_unique is False for 
output_file_path = './data/behavior_triplets_train90_filtered.txt'

counter = 0
with open(output_file_path, 'w') as output_file:
    for triplet in triplets_from_txt:
        words = list(concepts.iloc[triplet]["uniqueID"])
        if not not_unique(words):
            output_file.write(' '.join(triplet) + '\n')
        else: 
            counter+=1

print(f"Removed {counter} triplets")
            