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
import time
import httpx

MAX_RETRY_ATTEMPTS = 1
last_request_time = time.time()

# Martins key right now. Each triplet costs so lets think first before trying out many things
API_key = ""  # enter the key
os.environ["OPENAI_API_KEY"] = API_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/things-behavior-images/",
        help="Path to the directory containing the images",
    )
    parser.add_argument(
        "--train_fp",
        type=str,
        default="./data/100filtered_triplets.txt",
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
        default="./data/prompt_images.toml",
        help="Path to the file containing the prompt specifics for the images",
    )
    parser.add_argument(
        "--prompt_words_fp",
        type=str,
        default="./data/prompt_words.toml",
        help="Path to the file containing the prompt specifics for the words",
    )
    parser.add_argument(
        "--output_fp",
        type=str,
        default="./data/gpt4_triplet_output.pkl",
        help="Path to the file containing the output",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument(
        "--max_triplets",
        type=int,
        default=100,
        help="Max number of triplets to sample",
    )
    parser.add_argument(
        "--extract_responses",
        action="store_true",
        help="Extract responses",
    )

    return parser.parse_args()


def compute_updated_costs(n_images: int, output_tokens=1) -> None: 
    """Each image costs in low resolution processing 85 tokens.
    Price per 1k tokens = 0.01 input. output = 0.03 per 1k tokens. we only have 10 output tokens 
    expected output tokens depend on what is specified in prompt, currently asking for only position, so 1 token.
    1,000 tokens is about 750 words.
    """
    p_per_input_token = 0.01 / 1_000
    p_per_output_token = 0.03 / 1_000
    
    token_per_image_triplet = 85 * 3
    tokens_n_triplets = token_per_image_triplet * n_images
    word_prompt_tokens = 1000/750 * 99 # word to token conversion (aprox)
    image_prompt_tokens = 1000/750 * 186 

    total_word_input_tokens = 4 + word_prompt_tokens # 3 words are aprox 4 tokens
    total_image_input_tokens = tokens_n_triplets + image_prompt_tokens

    output_estimate = n_images * output_tokens * 2
    cost_n_triplets = (
        total_image_input_tokens * p_per_input_token +
        total_word_input_tokens * p_per_input_token 
        + output_estimate * p_per_output_token
    )

    print(f"Cost estimate for {n_images} image triplets: ${cost_n_triplets}")


def triplet_generator(
    image_dir: str,
    image_size: int,
    triplet_matrix: np.ndarray,
    concepts: pd.DataFrame,
    non_unique_words: set = None,
    seed: int = 0,
    check_word_ambiguity=True,
): # Generator[tuple[list[Image.Image], list[str], np.ndarray], None, None]
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
        shuffled_triplet = triplet.copy()
        np.random.shuffle(shuffled_triplet)
        words = list(map(get_cls_name, shuffled_triplet))
        images = list(map(load_image, words))
        
        # if we only sample behavior triplets for images, we want to include all and then
        # not check for word ambiguity 
        if check_word_ambiguity and not_unique(words):
            continue
        yield images, words, list(triplet), list(shuffled_triplet)


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

    content = []
    for image in image_triplet:
        image_encoding = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image)}",
                "detail": "low",  # NOTE I set this to low to save on tokens 
            },
        }
        
        text_encoding = {"type": "text", "text": text}
        # content.append(text_encoding)
        content.append(image_encoding)
    return content


def get_word_query(word_triplet, text):
    """Content for the GPT 4 word query"""
    word_triplet = [w.replace("_", " ") for w in word_triplet]
    task_string = ",".join(word_triplet) #"Execute this task for the following words\n" + ",".join(word_triplet)
    # text += task_string 
    content = [{"type": "text", "text": task_string}]
    return content

def check_rate_limit():
    elapsed_time = time.time() - last_request_time
    # check num requests
    # num input tokens
    # check if too many tokens / requests within elapsed time

def get_gpt_response(client, 
                     content, 
                     prompt,
                     temperature=1, 
                     max_tokens=300, 
                     retry_count=1, 
                     base_delay=10, 
                     max_delay=200): 
    
    """TODO: calculate rate limits before making requests 
    write function check_rate_limit for checking time since last request 
    and call it before API call 
    TODO make max_tokens as small as possible """

    try: 
        # check_rate_limit()
        seed = np.random.randint(1, 123456789) 
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "system", 
                       "content": prompt,},
                       {"role": "user",
                       "content": content}],
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            seed=seed,
        ) # TODO see if response contains rate limit info
        print(response)

    # handle option that its a tpm or rpd limit error
    except Exception as e:
        if "Error code: 429" in str(e) and retry_count < MAX_RETRY_ATTEMPTS: 
            # wait_time = int(http_err.response.headers.get('Retry-After', 20))
            retry_count += 1
            delay = min(base_delay * (2 ** retry_count), max_delay) #with base_delay of 10 seconds
            print(f"Retrying in {delay} seconds (Attempt {retry_count}/{MAX_RETRY_ATTEMPTS})")
            time.sleep(delay)
            return get_gpt_response(client, content, prompt, temperature, max_tokens, retry_count)  # Retry API request after waiting
        else:
            print(f"An error occured: {e}")
            response = None
        
    global last_request_time
    last_request_time = time.time()
    return response


def process_image_response(response):
    
    try:
        choice = response.choices[0].message.content # string 
        choice = choice.strip()  
        if len(choice) > 1:
            descriptions = choice.split("\n")[0]
            descriptions = [d.strip().replace(" ", "_").lower() for d in descriptions.split(",")]
            choice = choice.split("\n")[-1] 
            if len(choice) > 1:
                # if choice is still too long, extract the integer from the string
                choice = int(''.join(filter(str.isdigit, choice)))
        else:
            descriptions = None
        position = int(choice) - 1  # convert to 0-based indexing

    except Exception as e: 
        print(f"an error occured (images): {e}")
        position = None
        descriptions = None
    print("image response: ", choice)
    return position, descriptions


def process_word_response(response):
    
    try:
        choice = response.choices[0].message.content
        choice = choice.strip()
        if len(choice) > 1: 
            # assuming that descriptions are in the first line of response 
            # and position in the last, whats in between doesnt matter
            descriptions = choice.split("\n")[0]
            descriptions = [d.strip().replace(" ", "_").lower() for d in descriptions.split(",")]
            choice = choice.split("\n")[-1] 
            if len(choice) > 1:
                # Extract the integer from the string
                choice = int(''.join(filter(str.isdigit, choice)))
        else: 
            descriptions = None
        position = int(choice) - 1  # convert to 0-based indexing
    except Exception as e: 
        print(f"an error occured (words): {e}")
        position = None
        descriptions = None

    print("word response: ", choice)
    return position, descriptions


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
    shuffled_triplet,
    prompt_words,
    prompt_images,
    temperature,
):
    triplet_responses = dict()
    triplet_responses["human_triplet_indices"] = human_indices
    triplet_responses["human_ooo_index"] = human_indices[-1]
    triplet_responses["human_triplet_words"] = word_triplet
    print("word triplet: ", word_triplet)
    triplet_responses["human_ooo_word"] = word_triplet[-1]
    triplet_responses["errors"] = 0

    image_query = get_image_query(image_triplet, prompt_images["prompt_test"])
    word_query = get_word_query(word_triplet, prompt_words["prompt_test"])

    try: 
        image_response = get_gpt_response(client, prompt_images, image_query, temperature, max_tokens=300)
        word_response = get_gpt_response(client, prompt_words, word_query, temperature, max_tokens=300)

        image_position, image_description = process_image_response(image_response)
        word_position, word_description = process_word_response(word_response)

        # We ensure that the ooo index is always at the last position of the list
        ooo_image = move_ooo_to_last_position(shuffled_triplet, image_position)
        ooo_word = move_ooo_to_last_position(shuffled_triplet, word_position)

        ## Add below to the triplet_responses
        triplet_responses["shuffled_triplet"] = shuffled_triplet
        triplet_responses["gpt_image_triplet_indices"] = ooo_image
        triplet_responses["gpt_image_ooo"] = ooo_image[-1]
        triplet_responses["gpt_word_indices"] = ooo_word
        triplet_responses["gpt_word_ooo"] = ooo_word[-1]
        triplet_responses["image_position"] = image_position
        triplet_responses["word_position"] = word_position
        triplet_responses["image_description"] = image_description
        triplet_responses["word_description"] = word_description

    except Exception as e: 
        ooo_image = human_indices 
        ooo_word = human_indices
        
        triplet_responses["shuffled_triplet"] = shuffled_triplet
        triplet_responses["gpt_image_triplet_indices"] = ooo_image
        triplet_responses["gpt_image_ooo"] = None
        triplet_responses["gpt_word_indices"] = ooo_word
        triplet_responses["gpt_word_ooo"] = None
        triplet_responses["errors"] = 1
        triplet_responses["image_position"] = None
        triplet_responses["word_position"] = None
        triplet_responses["image_description"] = None
        triplet_responses["word_description"] = None

        print(f"An exception occured: {e}")

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
    for i, (image_triplet, word_triplet, human_indices, shuffled_triplet) in enumerate(dataloader):
        """We shuffle the triplet since we take the triplet indices from behavior, where we store
        the odd one out decision always in the last position and we want to make sure that the ordering
        is random and not related to the position of the odd one out as identified from human behavior"""
        print(f"Process triplet {i+1} / {args.max_triplets}")

        triplet_responses = process_triplet(
            client,
            image_triplet,
            word_triplet,
            human_indices,
            shuffled_triplet,
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
    client = OpenAI() # TODO check if new initialization
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
