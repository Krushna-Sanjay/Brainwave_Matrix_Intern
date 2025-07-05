import praw
import json
import re
import spacy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('vader_lexicon')

# Reddit API credentials
reddit = praw.Reddit(
    client_id='16RpLOg2E3X42jsLklKcTA',
    client_secret='NTvUkFkVbdszRBWtgYJLCEcOCvEFnQ',
    user_agent='script:ai_sentiment:v1.0 (by /u/Informal-Demand-6510)'
)

# Query setup
query = '"AI replacing jobs" OR "AI job automation" OR "fear of AI" OR "excitement about AI"'
posts = reddit.subreddit('all').search(query, sort='new', time_filter='month', limit=100)

# Load spaCy and VADER
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Load RoBERTa
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
labels = ["negative", "neutral", "positive"]

def get_roberta_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    return labels[np.argmax(probs)], max(probs)

# Process posts
cleaned = []
seen = set()

for post in posts:
    title = post.title.strip()
    body = post.selftext.strip()
    created_utc = post.created_utc

    combined_text = f"{title} {body}"

    if combined_text in seen:
        continue
    seen.add(combined_text)

    cleaned_text = re.sub(r"http\S+|[^A-Za-z\s]", "", combined_text).lower()

    doc = nlp(cleaned_text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
    if not lemmas:
        continue

    final_text = " ".join(lemmas)

    # VADER Sentiment
    vader_score = sia.polarity_scores(final_text)
    vader_sentiment = (
        "positive" if vader_score["compound"] >= 0.05 else
        "negative" if vader_score["compound"] <= -0.05 else
        "neutral"
    )

    # RoBERTa Sentiment
    try:
        roberta_sentiment, roberta_confidence = get_roberta_sentiment(final_text)
    except Exception as e:
        print(f"❌ BERT error: {e}")
        continue

    cleaned.append({
        "text": final_text,
        "timestamp": created_utc,
        "vader_compound": vader_score["compound"],
        "vader_sentiment": vader_sentiment,
        "bert_sentiment": roberta_sentiment,
        "bert_confidence": roberta_confidence
    })

# Save to JSON
with open("reddit_ai_sentiment.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2)

print(f"✅ {len(cleaned)} posts saved with both VADER and BERT sentiment to reddit_ai_sentiment.json")



# Load the saved JSON data
df = pd.read_json("reddit_ai_sentiment.json")

# Convert timestamps to datetime
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

# Create 5-day interval bins
df["date_bin"] = (df["datetime"].dt.floor("D") - pd.to_timedelta(df["datetime"].dt.dayofyear % 5, unit="D"))

# Consistent sentiment color palette
sentiment_palette = {"positive": "#66c2a5", "neutral": "#8da0cb", "negative": "#fc8d62"}

# --------- BERT Based Sentiment Trend ---------
from itertools import product

# Full date bins
all_bins = pd.date_range(start=df['date_bin'].min(), end=df['date_bin'].max(), freq='5D')
sentiments = ["positive", "neutral", "negative"]
bert_index = pd.DataFrame(product(all_bins, sentiments), columns=["date_bin", "bert_sentiment"])

# BERT grouped and filled
bert_counts = df.groupby(["date_bin", "bert_sentiment"]).size().reset_index(name="count")
bert_grouped = pd.merge(bert_index, bert_counts, on=["date_bin", "bert_sentiment"], how="left").fillna(0)
bert_grouped["count"] = bert_grouped["count"].astype(int)

plt.figure(figsize=(10, 6))
sns.lineplot(data=bert_grouped, x="date_bin", y="count", hue="bert_sentiment", marker="o", palette=sentiment_palette)

plt.title("Sentiment Trend on Reddit (RoBERTa)")
plt.xlabel("Date")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.grid(True)
plt.tight_layout()
plt.savefig("bert_sentiment_trend.png")
plt.show()

# --------- VADER Based Sentiment Trend ---------
vader_index = pd.DataFrame(product(all_bins, sentiments), columns=["date_bin", "vader_sentiment"])
vader_counts = df.groupby(["date_bin", "vader_sentiment"]).size().reset_index(name="count")
vader_grouped = pd.merge(vader_index, vader_counts, on=["date_bin", "vader_sentiment"], how="left").fillna(0)
vader_grouped["count"] = vader_grouped["count"].astype(int)

plt.figure(figsize=(10, 6))
sns.lineplot(data=vader_grouped, x="date_bin", y="count", hue="vader_sentiment", marker="o", palette=sentiment_palette)

plt.title("Sentiment Trend on Reddit (VADER)")
plt.xlabel("Date")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.grid(True)
plt.tight_layout()
plt.savefig("vader_sentiment_trend.png")
plt.show()

# Sentiment Distribution (Bar Plot)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# VADER Distribution
vader_counts = df["vader_sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
axs[0].bar(vader_counts.index, vader_counts.values, color=["#66c2a5", "#ffd92f", "#fc8d62"])
axs[0].set_title("VADER Sentiment Distribution")
axs[0].set_ylabel("Count")

# BERT Distribution
bert_counts = df["bert_sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
axs[1].bar(bert_counts.index, bert_counts.values, color=["#66c2a5", "#ffd92f", "#fc8d62"])
axs[1].set_title("RoBERTa Sentiment Distribution")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("sentiment_distribution_barplot.png")
plt.show()

# Comparison Heatmap (VADER vs BERT)
comparison = pd.crosstab(df["vader_sentiment"], df["bert_sentiment"])
sns.heatmap(comparison, annot=True, fmt="d", cmap="coolwarm")
plt.title("VADER vs RoBERTa Sentiment Comparison")
plt.xlabel("RoBERTa Sentiment")
plt.ylabel("VADER Sentiment")
plt.tight_layout()
plt.savefig("vader_vs_bert_heatmap.png")
plt.show()
