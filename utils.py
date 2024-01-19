import numpy as np
import pandas as pd

# TODO Don't return indices, but return the filtered concepts maybe just simply word
# list


def filter_concepts_cls_names(concept_path: str, word_path: str):
    """Filter the concepts to only include the 48 classes in the word list. Based on
    this we can then select the images to get the fully sampled triplet matrix"""

    def filter_cls_names(cls_names: np.ndarray):
        cls_names = cls_names[(cls_names != "camera") & (cls_names != "file")]
        cls_names = np.append(cls_names, ["camera1", "file1"])
        cls_names = [s.replace(" ", "_") for s in cls_names]
        return cls_names

    concepts = pd.read_csv(concept_path, sep="\t")
    words48 = pd.read_csv(word_path, encoding="utf-8")
    cls_names = words48["Word"].values
    cls_names = filter_cls_names(cls_names)

    regex_pattern = "|".join([rf"\b{s}\b" for s in cls_names])
    filtered_concepts = concepts[
        concepts["uniqueID"].str.contains(regex_pattern, regex=True)
    ]
    indices = filtered_concepts.index
    return indices, filtered_concepts


def parse_non_unique_words(path: str) -> list:
    """Parse the file and return a list of words"""
    file = pd.read_excel(path)
    non_unique = set(file[file["alternative"].notnull()]["unique_id"])
    return non_unique


if __name__ == "__main__":
    """Example code how to get the indices of the 48 classes"""
    concept_path = "./data/things_concepts.tsv"
    word_path = "./data/words48.csv"
    indices, filtered_concepts = filter_concepts_cls_names(concept_path, word_path)
