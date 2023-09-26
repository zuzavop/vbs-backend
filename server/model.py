import open_clip


loaded_model = None
loaded_tokenizer = None
loaded_preprocess = None


def load_open_clip():
    # load model and tokenizer via open clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    return model, tokenizer, preprocess


loaded_model, loaded_tokenizer, loaded_preprocess = load_open_clip()
