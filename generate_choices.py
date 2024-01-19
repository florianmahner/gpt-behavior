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
from utils import parse_non_unique_words


# Martins key right now. Each triplet costs so lets think first before trying out many things
API_key = None  # enter the key
os.environ["OPENAI_API_KEY"] = API_key


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
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument(
        "--max_triplets",
        type=int,
        default=1000,
        help="Max number of triplets to sample",
    )
    parser.add_argument(
        "--extract_responses",
        action="store_true",
        help="Extract responses",
    )

    return parser.parse_args()


def compute_updated_costs(n_images: int) -> None:
    """Each image costs in low resolution processing 85 tokens.
    Price per 1k tokens = 0.01 input. output = 0.03 per 1k tokens. we only have 10 output tokes
    """
    p_per_input_token = 0.01 / 1_000
    p_per_output_token = 0.03 / 1_000
    token_per_image_triplet = 85 * 3
    tokens_n_triplets = token_per_image_triplet * n_images

    output_estimate = n_images * 10
    cost_n_triplets = (
        tokens_n_triplets * p_per_input_token + output_estimate * p_per_output_token
    )
    print(f"Cost for {n_images} image triplets: ${cost_n_triplets}")


def triplet_generator(
    image_dir: str,
    image_size: int,
    triplet_matrix: np.ndarray,
    concepts: pd.DataFrame,
    non_unique_words: set = None,
    seed: int = 0,
    check_word_ambiguity=True,
) -> Generator[tuple[list[Image.Image], list[str], np.ndarray], None, None]:
    def get_cls_name(index: int) -> str:
        return concepts.iloc[index]["uniqueID"]

    def load_image(cls_name: str) -> Image:
        image_path = os.path.join(image_dir, cls_name + ".jpg")
        image = Image.open(image_path).resize((image_size, image_size))
        return image

    def not_unique(words: str) -> bool:
        """checks if the triplet contains non-unique words, e.g. words that are not twice
        in the things object images and therefore might be ambiguous for when collecting
        gpt3 ooo judgments based on words"""
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

        # if we only sample behavior triplets for images, we want to include all and then
        # not check for word ambiguity
        if check_word_ambiguity and not_unique(words):
            continue
        yield images, words, list(triplet)


def load_instructions(path: str) -> str:
    """Open the file and read the instructions, handling potential errors."""
    try:
        with open(path, "r") as file:
            return file.read()
    except OSError as e:
        print(f"Failed to open or read the file {path}. Error: {e}")
        raise OSError(e)


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
    generator = triplet_generator(
        image_fp, image_size, triplet_matrix, concepts, non_unique_words, seed
    )

    return generator


def save_responses(responses: list, metadata: dict, output_fp: str) -> None:
    df = pd.DataFrame(responses, columns=responses[0].keys())
    save_dict = {"metadata": metadata, "dataframe": df}
    with open(output_fp, "wb") as file:
        pickle.dump(save_dict, file)


def pop_keys_from_arguments(metadata: dict) -> dict:
    metadata = copy.deepcopy(vars(args))
    for key in [
        "config",
        "root_table",
        "max_triplets",
        "output_fp",
    ]:
        metadata.pop(key)
    return metadata


def move_ooo_to_last_position(human_indices, ooo_index):
    indices = copy.copy(human_indices)
    ooo = indices.pop(ooo_index)
    indices.append(ooo)
    return indices


def process_triplet(
    client,
    image_triplet,
    word_triplet,
    human_indices,
    prompt_words,
    prompt_images,
    temperature,
):
    triplet_responses = dict()
    triplet_responses["human_triplet_indices"] = human_indices
    triplet_responses["human_ooo_index"] = human_indices[-1]
    triplet_responses["human_triplet_words"] = word_triplet
    triplet_responses["human_ooo_word"] = word_triplet[-1]

    image_query = get_image_query(image_triplet, prompt_images["prompt"])
    word_query = get_word_query(word_triplet, prompt_words["prompt"])

    image_response = get_gpt_response(client, image_query, temperature, max_tokens=300)
    word_response = get_gpt_response(client, word_query, temperature, max_tokens=300)

    image_descriptions, image_choice, image_position = process_image_response(
        image_response
    )
    word_choice, word_position = process_word_response(word_response)

    # We ensure that the ooo index is always at the last position of the list
    ooo_image = move_ooo_to_last_position(human_indices, image_position)
    ooo_word = move_ooo_to_last_position(human_indices, word_position)

    ## Add below to the triplet_responses
    triplet_responses["gpt_image_descriptions"] = image_descriptions
    triplet_responses["gpt_image_word_choice"] = image_choice
    triplet_responses["gpt_image_triplet_indices"] = ooo_image
    triplet_responses["gpt_image_ooo"] = ooo_image[-1]
    triplet_responses["gpt_image_word_choice"] = image_choice
    triplet_responses["gpt_word_indices"] = ooo_word
    triplet_responses["gpt_word_ooo"] = ooo_word[-1]
    triplet_responses["gpt_word_word_choice"] = word_choice
    return triplet_responses


def extract_responses(
    dataloader,
    client,
    metadata,
    prompt_images,
    prompt_words,
    max_triplets,
    output_fp,
    temperature,
):
    responses = []
    for i, (image_triplet, word_triplet, human_indices) in enumerate(dataloader):
        # We shuffle the triplet since we take the triplet indices from behavior, where we store
        # the odd one out decision always in the last position and we want to make sure that the ordering
        # is random and not related to the position of the odd one out as identified from human behavior
        print(f"Process triplet {i} / {args.max_triplets}")

        triplet_responses = process_triplet(
            client,
            image_triplet,
            word_triplet,
            human_indices,
            prompt_words,
            prompt_images,
            temperature,
        )

        responses.append(triplet_responses)
        save_responses(responses, metadata, output_fp)

        if i == max_triplets - 1:
            break


def main(args):
    compute_updated_costs(args.max_triplets)

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
    metadata = pop_keys_from_arguments(vars(args))

    if args.extract_responses:
        extract_responses(
            dataloader,
            client,
            metadata,
            prompt_images,
            prompt_words,
            args.max_triplets,
            args.output_fp,
            args.temperature,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
