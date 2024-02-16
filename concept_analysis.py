""" functions from Flo for loading and fetching the 48 objects from the things concepts 
"""
import numpy as np 
import pandas as pd

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

concept_path = "./data/misc/things_concepts.tsv"
concepts = load_concepts(concept_path)

# Concepts has a column of words. i want to find the indices of the words that are in the cls_names
# and then sort the rsm by those indices.
indices_48 = filter_concepts_cls_names(concepts, cls_names)