# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse

import os
import torch

import open_clip

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification


# Load all available noun lists in the selected directory
def load_noun_lists(path):
    noun_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

    noun_lists = []
    for noun_file in noun_files:
        with open(noun_file) as f:
            nouns = [line.strip() for line in f]
        noun_lists.append([os.path.basename(noun_file), nouns])

    return noun_lists


# Scale normalization for features
def normalize(features):
    return features / features.norm(dim=-1, keepdim=True)


# Create LAION embeddings for the noun lists
def load_laion(path):
    noun_lists = load_noun_lists(path)
    for noun_list in noun_lists:
        file_name, noun_list = noun_list
        result_path = os.path.join(path, f'{file_name[:-4]}-laion.ptt')

        if not os.path.exists(result_path):
            print(f'Process noun list: {file_name}')

            # Load model and tokenizer via open clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
            )
            tokenizer = open_clip.get_tokenizer(
                'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
            )

            tokenized_noun_list = [tokenizer(f'a photo of {n}') for n in noun_list]
            cat_tokenized_noun_list = torch.cat(tokenized_noun_list)
            with torch.no_grad():
                embedded_noun_list = model.encode_text(cat_tokenized_noun_list)
            norm_embedded_noun_list = normalize(embedded_noun_list)

            torch.save(norm_embedded_noun_list, result_path)

            print(f'Noun list processed: {file_name}')
        else:
            print(f'Noun list already processed: {file_name}')


# Create Open Clip embeddings for the noun lists
def load_open_clip(path):
    noun_lists = load_noun_lists(path)
    for noun_list in noun_lists:
        file_name, noun_list = noun_list
        result_path = os.path.join(path, f'{file_name[:-4]}-clip.ptt')

        if not os.path.exists(result_path):
            print(f'Process noun list: {file_name}')

            # Load model and tokenizer via transformers
            processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
            model = AutoModelForZeroShotImageClassification.from_pretrained(
                'openai/clip-vit-large-patch14'
            )

            inputs = processor(
                text=[f'a photo of {n}' for n in noun_list],
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                embedded_noun_list = model(**inputs).text_embeds
            norm_embedded_noun_list = normalize(embedded_noun_list)

            torch.save(norm_embedded_noun_list, result_path)

            print(f'Noun list processed: {file_name}')
        else:
            print(f'Noun list already processed: {file_name}')


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Open noun lists and convert the nouns to embeddings.'
    )

    # Define the arguments
    parser.add_argument('--base-dir', required=True, help='Base data directory')
    parser.add_argument('--model-name', required=True, help='Model name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments and call the main function
    base_data_dir = args.base_dir
    model_name = args.model_name

    if 'clip-laion' in model_name:
        load_laion(base_data_dir)
    if 'clip-openai' in model_name:
        load_open_clip(base_data_dir)
