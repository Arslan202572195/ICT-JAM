from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}


print("Loading sentiment model...")

sentiment_pipeline = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1,  # CPU
)

print("Sentiment model loaded successfully.")


def clean_label(model_label: str) -> str:
    """
    Convert the model's internal label into a user-friendly label.
    """

    return LABEL_MAP.get(
        model_label,
        model_label.title(),
    )


def analyse_sentiment(text: str) -> dict:
    """
    Analyse one survey comment.

    Returns:
        {
            "sentiment": "Positive",
            "confidence": 0.9123
        }
    """

    if not text or not text.strip():
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
        }

    result = sentiment_pipeline(
        text.strip(),
        truncation=True,
        max_length=512,
    )[0]

    return {
        "sentiment": clean_label(result["label"]),
        "confidence": round(float(result["score"]), 4),
    }


def analyse_sentiment_batch(comments: list[str]) -> list"""
    Analyse multiple comments in one model operation.

    Batch processing is generally more efficient than calling
    the model separately for every comment.
    """

    valid_comments = [
        comment.strip()
        for comment in comments
        if comment and comment.strip()
    ]

    if not valid_comments:
        return []

    model_results = sentiment_pipeline(
        valid_comments,
        truncation=True,
        max_length=512,
        batch_size=8,
    )

    processed_results = []

    for comment, result in zip(valid_comments, model_results):
        processed_results.append({
            "feedback": comment,
            "sentiment": clean_label(result["label"]),
            "confidence": round(float(result["score"]), 4),
        })

    return processed_results