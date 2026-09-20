from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    device=-1,
)

print("Sentiment model loaded successfully.")


app = FastAPI(
    title="Student Survey Analysis API",
    description=(
        "An API for analysing end-of-semester student survey "
        "comments using a locally hosted Transformer model."
    ),
    version="1.0.0",
)


class SurveyRequest(BaseModel):
    feedback: str = Field(
        min_length=1,
        max_length=5000,
        description="The student survey comment to analyse.",
        examples=[
            "The lecturer sucks."
        ],
    )


class BatchSurveyRequest(BaseModel):
    comments: list[str] = Field(
        min_length=1,
        max_length=500,
        description="A list of student survey comments.",
        examples=[
            [
                "The lecturer explains the subject clearly.",
                "The WiFi connection is extremely unreliable.",
                "The timetable is acceptable.",
            ]
        ],
    )


class SentimentResponse(BaseModel):
    feedback: str
    sentiment: str
    confidence: float


class SentimentCounts(BaseModel):
    positive: int
    neutral: int
    negative: int


class SentimentPercentages(BaseModel):
    positive: float
    neutral: float
    negative: float


class DashboardSummaryResponse(BaseModel):
    total_responses: int
    sentiment_counts: SentimentCounts
    sentiment_percentages: SentimentPercentages
    processed_responses: list[SentimentResponse]


def clean_label(model_label):
    return LABEL_MAP.get(
        model_label,
        str(model_label).title(),
    )


def get_valid_comments(comments):
    return [
        comment.strip()
        for comment in comments
        if comment and comment.strip()
    ]


def analyse_sentiment(text):
    cleaned_text = text.strip()

    if not cleaned_text:
        return {
            "feedback": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
        }

    result = sentiment_pipeline(
        cleaned_text,
        truncation=True,
        max_length=512,
    )[0]

    return {
        "feedback": cleaned_text,
        "sentiment": clean_label(result["label"]),
        "confidence": round(float(result["score"]), 4),
    }


def analyse_sentiment_batch(comments):
    valid_comments = get_valid_comments(comments)

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
        processed_results.append(
            {
                "feedback": comment,
                "sentiment": clean_label(result["label"]),
                "confidence": round(float(result["score"]), 4),
            }
        )

    return processed_results


@app.get("/", tags=["System"])
def home():
    return {
        "message": "Student Survey Analysis API is running.",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "processing_location": "Local computer",
        "processing_device": "CPU",
    }


@app.post(
    "/analyse/sentiment",
    response_model=SentimentResponse,
    tags=["Sentiment Analysis"],
)
def analyse_single_comment(request: SurveyRequest):
    try:
        return analyse_sentiment(request.feedback)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Sentiment analysis failed: " + str(error),
        ) from error


@app.post(
    "/analyse/batch",
    response_model=list[SentimentResponse],
    tags=["Sentiment Analysis"],
)
def analyse_multiple_comments(request: BatchSurveyRequest):
    try:
        valid_comments = get_valid_comments(request.comments)

        if not valid_comments:
            raise HTTPException(
                status_code=400,
                detail="At least one non-empty comment is required.",
            )

        return analyse_sentiment_batch(valid_comments)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Batch sentiment analysis failed: " + str(error),
        ) from error


@app.post(
    "/analytics/summary",
    response_model=DashboardSummaryResponse,
    tags=["Dashboard Analytics"],
)
def create_dashboard_summary(request: BatchSurveyRequest):
    try:
        valid_comments = get_valid_comments(request.comments)

        if not valid_comments:
            raise HTTPException(
                status_code=400,
                detail="At least one non-empty comment is required.",
            )

        processed_results = analyse_sentiment_batch(valid_comments)

        counts = {
            "Positive": 0,
            "Neutral": 0,
            "Negative": 0,
        }

        for result in processed_results:
            sentiment = result["sentiment"]

            if sentiment in counts:
                counts[sentiment] += 1

        total_responses = len(processed_results)

        if total_responses == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid comments were processed.",
            )

        positive_percentage = round(
            counts["Positive"] / total_responses * 100,
            2,
        )

        neutral_percentage = round(
            counts["Neutral"] / total_responses * 100,
            2,
        )

        negative_percentage = round(
            counts["Negative"] / total_responses * 100,
            2,
        )

        return {
            "total_responses": total_responses,
            "sentiment_counts": {
                "positive": counts["Positive"],
                "neutral": counts["Neutral"],
                "negative": counts["Negative"],
            },
            "sentiment_percentages": {
                "positive": positive_percentage,
                "neutral": neutral_percentage,
                "negative": negative_percentage,
            },
            "processed_responses": processed_results,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Dashboard analysis failed: " + str(error),
        ) from error