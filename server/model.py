import open_clip

import configs as c


loaded_model = None
loaded_tokenizer = None
loaded_preprocess = None


# Load functions for the open clip LAION model
def load_laion():
    # Load model and tokenizer via open clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    return model, tokenizer, preprocess


# Selector for the load model
load_function = load_laion
if c.MODEL == 'laion':
    load_function = load_laion

loaded_model, loaded_tokenizer, loaded_preprocess = load_function()
