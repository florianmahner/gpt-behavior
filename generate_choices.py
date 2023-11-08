import numpy as np
import pandas as pd
import os
from PIL import Image
import base64
import requests
import io
import toml
import random
import openai


def parse_non_unique_words(path: str) -> list:
    """Parse the file and return a list of words"""
    file = pd.read_excel(path)
    # If an alternativ is there then it is not unique
    non_unique = list(file[file["alternative"].notnull()]["unique_id"])
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
    train_triplet: str,
    concepts: pd.DataFrame,
    n: int = 10,
    non_unique_words: list = None,
):
    image_triplets = np.loadtxt(train_triplet, dtype=int)
    random.seed(0)
    random.shuffle(image_triplets)
    image_triplets = image_triplets[:n]

    def get_cls_name(index: int) -> str:
        return concepts.iloc[index]["uniqueID"]

    def load_image(cls_name: str) -> Image:
        image_path = os.path.join(image_dir, cls_name + ".jpg")
        image = Image.open(image_path).resize((256, 256))
        return image

    for triplet in image_triplets:
        words = list(map(get_cls_name, triplet))
        images = list(map(load_image, words))

        for w in words:
            if w in non_unique_words:
                continue

        yield images, words


def load_instructions(path: str):
    """Open the file and read the instructions and catch errors"""
    try:
        with open(path, "r") as f:
            instructions = f.read()
        return instructions
    except OSError as e:
        print(e)


def get_image_response(image_triplet, payload, headers):
    combined_image = np.concatenate(
        [np.array(image) for image in image_triplet], axis=1
    )

    # This below is only for images. What should be the prompt for text?
    payload["messages"][0]["content"].append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(Image.fromarray(combined_image))}",
                "detail": "low",
            },
        }
    )

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()
    return response


def get_text_response(word_triplet, payload, headers):
    word_triplet = [w.replace("_", " ") for w in word_triplet]
    task_string = "Execute this task for the following words\n" + " ".join(word_triplet)

    payload["messages"][0]["content"][0][
        "text"
    ] += task_string  # add triplet of words to the prompt

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()

    return response


def main():
    image_dir = "./data/images/"
    train_fp = "./data/trainset.txt"
    things_concepts = pd.read_csv("./data/things_concepts.tsv", sep="\t")
    non_unique_words = parse_non_unique_words("./data/NonUniqueWords_edited.xlsx")

    prompt_images = toml.load("./data/prompt_specifics_images.toml")
    payload_images = prompt_images["payload"]
    headers_images = prompt_images["headers"]

    prompt_words = toml.load("./data/prompt_specifics_words.toml")
    payload_words = prompt_words["payload"]
    headers_words = prompt_words["headers"]

    dl = dataloader(image_dir, train_fp, things_concepts, 1000, non_unique_words)

    for image_tripet, word_triplet in dl:
        # get_image_response(image_tripet, payload_images, headers_images)
        get_text_response(word_triplet, payload_words, headers_words)


if __name__ == "__main__":
    main()
