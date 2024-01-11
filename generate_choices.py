import numpy as np
import pandas as pd
import os
from PIL import Image
import base64
import io
import toml
import random
import pickle
import copy
from typing import Generator
from openai import OpenAI
from tomlparse import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/images/",
        help="Path to the directory containing the images",
    )
    parser.add_argument(
        "--train_fp",
        type=str,
        default="./data/trainset.txt",
        help="Path to the file containing the image triplets",
    )
    parser.add_argument(
        "--concepts_fp",
        type=str,
        default="./data/things_concepts.tsv",
        help="Path to the file containing the concepts",
    )
    parser.add_argument(
        "--non_unique_words_fp",
        type=str,
        default="./data/NonUniqueWords_edited.xlsx",
        help="Path to the file containing the non-unique words",
    )
    parser.add_argument(
        "--prompt_images_fp",
        type=str,
        default="./data/prompt_specifics_images.toml",
        help="Path to the file containing the prompt specifics for the images",
    )
    parser.add_argument(
        "--prompt_words_fp",
        type=str,
        default="./data/prompt_specifics_words.toml",
        help="Path to the file containing the prompt specifics for the words",
    )
    parser.add_argument(
        "--output_fp",
        type=str,
        default="./data/gpt4_choice_words_images.pkl",
        help="Path to the file containing the output",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument(
        "--max_triplets",
        type=int,
        default=1000,
        help="Max number of triplets to sample",
    )

    return parser.parse_args()


def parse_non_unique_words(path: str) -> list:
    """Parse the file and return a list of words"""
    file = pd.read_excel(path)
    non_unique = set(file[file["alternative"].notnull()]["unique_id"])
    return non_unique


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000) * 0.01 + (completion_tokens / 1000) * 0.03


def dataloader(
    image_dir: str,
    image_size: int,
    triplet_matrix: np.ndarray,
    concepts: pd.DataFrame,
    non_unique_words: set = None,
    seed: int = 0,
) -> Generator[tuple[list[Image.Image], list[str], np.ndarray], None, None]:
    def get_cls_name(index: int) -> str:
        return concepts.iloc[index]["uniqueID"]

    def load_image(cls_name: str) -> Image:
        image_path = os.path.join(image_dir, cls_name + ".jpg")
        image = Image.open(image_path).resize((image_size, image_size))
        return image

    def not_unique(words: str) -> bool:
        """check if the triplet contains non-unique words, e.g. words that are not twice
        in the things object images and therefore might be ambiguous for when collecting
        gpt3 ooo. judgments based on words"""
        words = set(words)
        n = len(words.intersection(non_unique_words))
        return True if n > 0 else False

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        return rng

    rng = set_seed(seed)
    rng.shuffle(triplet_matrix, axis=0)
    for triplet in triplet_matrix:
        words = list(map(get_cls_name, triplet))
        images = list(map(load_image, words))

        if not_unique(words):
            continue
        yield images, words, triplet


def load_instructions(path: str):
    """Open the file and read the instructions and catch errors"""
    try:
        with open(path, "r") as f:
            instructions = f.read()
        return instructions
    except OSError as e:
        print(e)


def get_image_query(image_triplet, text):
    """Content for the GPT 4 image query"""

    def encode_image(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        byte_data = buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        return base64_str.decode("utf-8")

    content = copy.deepcopy(text)
    content = [{"type": "text", "text": text}]
    for image in image_triplet:
        image_encoding = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image)}",
                "detail": "low",  # NOTE I set this to low to save on tokens
            },
        }
        content.append(image_encoding)
    return content


def get_word_query(word_triplet, text):
    """Content for the GPT 4 word query"""
    word_triplet = [w.replace("_", " ") for w in word_triplet]
    task_string = "Execute this task for the following words\n" + ",".join(word_triplet)
    text += task_string
    content = [{"type": "text", "text": text}]
    return content


def get_gpt_response(client, content, temperature=0.5, max_tokens=300):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
    )

    return response


def process_image_response(response):
    choice = response.choices[0].message.content
    descriptions, choice = choice.split("\n")
    descriptions = [
        d.strip().replace(" ", "_").lower() for d in descriptions.split(",")
    ]

    choice_name, position = choice.split(",")
    choice_name = choice_name.replace(" ", "_")
    position = int(position) - 1  # convert to 0-based indexing

    return descriptions, choice_name, position


def process_word_response(response):
    choice = response.choices[0].message.content
    choice, position = choice.split(",")
    position = int(position) - 1  # convert to 0-based indexing
    choice = choice.replace(" ", "_")
    return choice, position


def shuffle_data(images: list, words: list, triplet: list):
    zipped = list(zip(images, words, triplet))
    random.shuffle(zipped)
    images, words, indices = zip(*zipped)
    words, indices = list(words), list(indices)
    return images, words, indices


def create_dataloader(
    image_fp, triplet_fp, concept_fp, non_unique_words_fp, image_size, seed=0
):
    """Create a dataloader of the images, words and indices used to probe gpt4"""
    triplet_matrix = np.loadtxt(triplet_fp, dtype=int)
    concepts = pd.read_csv(concept_fp, sep="\t")
    non_unique_words = parse_non_unique_words(non_unique_words_fp)
    dl = dataloader(
        image_fp, image_size, triplet_matrix, concepts, non_unique_words, seed
    )

    return dl


# TODO Change all the stupid numpy array and list conver
# TODO calculate estimated cost for 1000 triplets
def main(args):
    dataloader = create_dataloader(
        args.image_dir,
        args.train_fp,
        args.concepts_fp,
        args.non_unique_words_fp,
        args.image_size,
        args.seed,
    )

    prompt_images = toml.load(args.prompt_images_fp)
    prompt_words = toml.load(args.prompt_words_fp)
    client = OpenAI()

    metadata = copy.deepcopy(vars(args))
    for key in [
        "config",
        "root_table",
        "max_triplets",
        "output_fp",
    ]:
        metadata.pop(key)

    responses = []
    for i, (image_triplet, word_triplet, human_indices) in enumerate(dataloader):
        # We shuffle the triplet since we take the triplet indices from behavior, where we store
        # the odd one out decision always in the last position and we want to make sure that the ordering
        # is random and not related to the position of the odd one out as identified from human behavior
        print(f"Process triplet {i} / {args.max_triplets}")

        triplet_responses = dict()
        triplet_responses["human_indices"] = list(human_indices)
        triplet_responses["human_triplet"] = word_triplet
        triplet_responses["human_choice"] = word_triplet[-1]
        triplet_responses["human_ooo"] = human_indices[-1]

        image_triplet, word_triplet, gpt_indices = shuffle_data(
            image_triplet, word_triplet, human_indices
        )

        image_query = get_image_query(image_triplet, prompt_images["prompt"])
        word_query = get_word_query(word_triplet, prompt_words["prompt"])

        image_response = get_gpt_response(
            client, image_query, args.temperature, max_tokens=300
        )
        word_response = get_gpt_response(
            client, word_query, args.temperature, max_tokens=300
        )

        image_descriptions, image_choice, image_position = process_image_response(
            image_response
        )
        word_choice, word_position = process_word_response(word_response)

        # We know move the index of the odd one out to the last position
        ooo_word = list(np.roll(gpt_indices, -word_position))
        ooo_image = list(np.roll(gpt_indices, -image_position))

        ## Add below to the triplet_responses
        triplet_responses["gpt_indices"] = gpt_indices
        triplet_responses["gpt_image_descriptions"] = image_descriptions
        triplet_responses["gpt_image_indices"] = ooo_image
        triplet_responses["gpt_image_ooo"] = ooo_image[-1]
        triplet_responses["gpt_image_choice"] = image_choice
        triplet_responses["gpt_word_indices"] = ooo_word
        triplet_responses["gpt_word_ooo"] = ooo_word[-1]
        triplet_responses["gpt_word_choice"] = word_choice

        responses.append(triplet_responses)
        df = pd.DataFrame(responses, columns=triplet_responses.keys())

        save_dict = {"metadata": metadata, "dataframe": df}
        with open(args.output_fp, "wb") as file:
            pickle.dump(save_dict, file)

        if i == args.max_triplets - 1:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
