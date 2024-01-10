import numpy as np
import pandas as pd
import os
from PIL import Image
import base64
import requests
import io
import toml
import random
import copy
from typing import Generator
from openai import OpenAI


def parse_non_unique_words(path: str) -> list:
    """Parse the file and return a list of words"""
    file = pd.read_excel(path)
    non_unique = set(file[file["alternative"].notnull()]["unique_id"])
    return non_unique


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    byte_data = buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode("utf-8")


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000) * 0.01 + (completion_tokens / 1000) * 0.03


def dataloader(
    image_dir: str,
    triplet_matrix: np.ndarray,
    concepts: pd.DataFrame,
    non_unique_words: set = None,
) -> Generator[tuple[list[Image.Image], list[str], np.ndarray], None, None]:
    def get_cls_name(index: int) -> str:
        return concepts.iloc[index]["uniqueID"]

    def load_image(cls_name: str) -> Image:
        image_path = os.path.join(image_dir, cls_name + ".jpg")
        image = Image.open(image_path).resize((64, 64))
        return image

    def not_unique(words: str) -> bool:
        """check if the triplet contains non-unique words, e.g. words that are not twice
        in the things object images and therefore might be ambiguous for when collecting
        gpt3 ooo. judgments based on words"""
        words = set(words)
        n = len(words.intersection(non_unique_words))
        return True if n > 0 else False

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


def get_image_choice(client, image_triplet, text):
    # We need to encode the images as base64 strings
    content = copy.deepcopy(text)
    content = [{"type": "text", "text": text}]

    for image in image_triplet:
        image_encoding = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{encode_image(image)}",
        }
        content.append(image_encoding)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": content}],
        max_tokens=300,
    )

    choice = response.choices[0].message.content
    descriptions, choice, position = process_image_response(response)
    return descriptions, choice, position


def process_image_response(response):
    choice = response.choices[0].message.content
    descriptions, choice = choice.split("\n")
    descriptions = descriptions.split(",")
    for i, d in enumerate(descriptions):
        d = d.strip()
        d = d.replace(" ", "_")
        d = d.lower()
        descriptions[i] = d

    choice_name, position = choice.split(",")
    position = int(position) - 1  # convert to 0-based indexing
    return descriptions, choice_name, position


def get_word_choice(client, word_triplet, text):
    word_triplet = [w.replace("_", " ") for w in word_triplet]
    task_string = "Execute this task for the following words\n" + ",".join(word_triplet)
    text += task_string
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            }
        ],
        max_tokens=300,
    )
    choice = response.choices[0].message.content
    choice, position = choice.split(",")
    position = int(position) - 1  # convert to 0-based indexing

    return choice, position


def shuffle_data(images: list, words: list, triplet: list):
    zipped = list(zip(images, words, triplet))
    random.shuffle(zipped)
    images, words, indices = zip(*zipped)
    words, indices = list(words), list(indices)
    return images, words, indices


# TODO Still need to figure out why choices are sometimes empty, e.g. why the GPT responses in not correctly parsed.
# NOTE sometimes I get error message when some input is not allowed...
# TODO Get it to run first and then outsource the specifics in the TOML file maybe!
# TODO Change all the stupid numpy array and list conver
def main():
    image_dir = "./data/images/"
    train_fp = "./data/trainset.txt"
    things_concepts = pd.read_csv("./data/things_concepts.tsv", sep="\t")
    non_unique_words = parse_non_unique_words("./data/NonUniqueWords_edited.xlsx")
    prompt_images = toml.load("./data/prompt_specifics_images.toml")
    prompt_words = toml.load("./data/prompt_specifics_words.toml")

    image_triplets = np.loadtxt(train_fp, dtype=int)

    random.seed(0)
    rng = np.random.default_rng(0)
    rng.shuffle(image_triplets, axis=0)

    client = OpenAI()
    dl = dataloader(image_dir, image_triplets, things_concepts, non_unique_words)
    max_iterations = 20
    responses = []
    i = 0
    for image_triplet, word_triplet, human_indices in dl:
        # try:
        # We shuffle the triplet since we take the triplet indices from behavior, where we store
        # the odd one out decision always in the last position and we want to make sure that the ordering
        # is random and not related to the position of the odd one out as identified from human behavior
        print("Process triplet", i)
        i += 1
        # if i != 8:
        #     continue

        triplet_responses = dict()
        triplet_responses["human_indices"] = list(human_indices)
        triplet_responses["human_triplet"] = word_triplet
        triplet_responses["human_choice"] = word_triplet[-1]
        triplet_responses["human_ooo"] = human_indices[-1]

        image_triplet, word_triplet, gpt_indices = shuffle_data(
            image_triplet, word_triplet, human_indices
        )

        (
            image_descriptions,
            image_choice,
            image_position,
        ) = get_image_choice(client, image_triplet, prompt_images["prompt"])

        word_choice, word_position = get_word_choice(
            client, word_triplet, prompt_words["prompt"]
        )

        word_choice = word_choice.replace(" ", "_")
        image_choice = image_choice.replace(" ", "_")

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

        print(word_triplet, image_descriptions)

        responses.append(triplet_responses)
        df = pd.DataFrame(responses, columns=triplet_responses.keys())
        df.to_pickle("./data/gpt4_choice_words_images.pkl")

        # except Exception as e:
        #     print(e)
        #     continue

        if i == max_iterations:
            break


if __name__ == "__main__":
    main()
