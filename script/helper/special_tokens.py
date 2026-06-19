SPECIAL_TOKENS = [
    "<region>",
    "</region>",
    "<object>",
    "</object>",
    "<class_id>",
    "</class_id>",
    "<color>",
    "</color>",
    "<evidence>",
    "</evidence>",
    "<bbox>",
    "</bbox>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<SEG>",
]

SEG_TOKEN = "<SEG>"


def resolve_tokenizer(tokenizer=None, processor=None):
    if processor is not None and hasattr(processor, "tokenizer"):
        return processor.tokenizer
    return tokenizer


def register_special_tokens(
    model,
    tokenizer=None,
    processor=None,
    special_tokens=None,
    seg_token=SEG_TOKEN,
):
    tokenizer = resolve_tokenizer(tokenizer=tokenizer, processor=processor)
    tokens = list(special_tokens or SPECIAL_TOKENS)

    if seg_token not in tokens:
        tokens.append(seg_token)

    try:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    if added > 0 and model is not None:
        model.resize_token_embeddings(len(tokenizer))

    if processor is not None and hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
    return model, tokenizer, processor, seg_token_id
