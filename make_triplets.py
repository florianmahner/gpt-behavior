"""make triplets from the given words"""
import pandas as pd
from itertools import combinations
import numpy as np 
import random

def get_indices(words_fp, concepts_fp):

    words = pd.read_csv(words_fp)
    concepts = pd.read_csv(concepts_fp, sep="\t")

    word_indices = []

    for _, row in words.iterrows():
        word = row['Word'].strip()

        # Check if the word is present in concepts
        if word in concepts['Word'].values:
            lookup_index = concepts[concepts['Word'] == word].index[0]
            word_indices.append(lookup_index)
        else:
            print(f"could not find {word} in concepts")
            word_indices.append(None)
    
    result_df = pd.DataFrame({'Index': word_indices})
    result_df.to_csv('word_indices.csv', index=False)

    return word_indices


def generate_triplets(words_fp, output_file):

    concept_fp = "./gpt-behavior/data/things_concepts.tsv"
    indices = get_indices(words_fp, concept_fp)

    # Generate all possible triplet combinations
    triplets = list(combinations(indices, 3))

    # Shuffle the order of lines (triplets)
    random.shuffle(triplets)

    # Write the triplets to a new file
    with open(output_file, 'w') as output:
        for triplet in triplets:
            # Shuffle the order of elements within each line (triplet)
            shuffled_triplet = ' '.join(map(str, random.sample(triplet, len(triplet))))
            output.write(shuffled_triplet + '\n')

word_list_path = './gpt-behavior/48_unambiguous_words.csv'
output_file_path = 'triplets_from_48.txt'

generate_triplets(word_list_path, output_file_path)

#########################################################################
"""functions from Flo for loading and fetching objects from concepts"""

def load_concepts(path: str = "./things_concepts.tsv") -> pd.DataFrame:
    concepts = pd.read_csv(path, encoding="utf-8", sep="\t")
    return concepts

def filter_concepts_cls_names(concepts, cls_names):
    """Filter by cls names"""
    cls_names = cls_names[(cls_names != "camera") & (cls_names != "file")]
    cls_names = np.append(cls_names, ["camera1", "file1"])
    # Replace white space with underscore in cls_names
    cls_names = [s.replace(" ", "_") for s in cls_names]
    regex_pattern = "|".join([rf"\b{s}\b" for s in cls_names])
    filtered_concepts = concepts[
            concepts["uniqueID"].str.contains(regex_pattern, regex=True)
        ]
    indices = filtered_concepts.index
    return indices