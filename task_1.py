import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text)         # remove extra whitespace
    return text.strip()


reviews = [
    "The food was tasty but the service was really slow 😐. I loved the dessert though 😋.",
    "Ambience was great 😍 but the main course was disappointing and lacked flavor 😒.",
    "The staff was friendly 🙂, but the seating area felt cramped and uncomfortable 😕.",
    "The pasta was delicious 😋, but the garlic bread was burnt and hard to eat 😞.",
    "I liked the overall vibe of the place 😊, but the prices were too high for the portion size 😐.",
    "The biryani was amazing 🔥, but the wait time was more than 40 minutes 😒.",
    "The drinks were refreshing 😍, but the food arrived cold 👎 which ruined the experience.",
    "Great music and atmosphere 🎶, but the service felt unorganized and slow 😕.",
    "The dessert was fantastic 🍰❤️, but the main dishes tasted bland 😐.",
    "The staff behaved nicely 🙂, but the order got mixed up twice 😡 before we finally got our food."
]

df = pd.DataFrame(reviews, columns=["Original_Review"])
df["Cleaned_Review"] = df["Original_Review"].apply(clean_text)

# Initialize VADER (Ensure NLTK is set up for this to work)
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # TextBlob Scores
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # VADER Scores
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores['compound']

    # Sentiment Label Logic (Based on TextBlob Polarity)
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series([polarity, subjectivity, vader_compound, sentiment])

# APPLYING NEW COLUMN NAMES
df[["TextBlob_Polarity", "TextBlob_Subjectivity", "VADER_Score", "TextBlob_Sentiment_Label"]] = df["Cleaned_Review"].apply(analyze_sentiment)

# Define a simple confidence metric based on absolute polarity and compound score
def compute_confidence(row):
    # Using the new column names here
    return round((abs(row['TextBlob_Polarity']) + abs(row['VADER_Score'])) / 2, 2)

# APPLYING NEW COLUMN NAME
df["Model_Confidence"] = df.apply(compute_confidence, axis=1)

# Confidence Score Distribution
plt.figure(figsize=(6,4))
# Using the new column name here
sns.histplot(df["Model_Confidence"], bins=5, kde=True, color='green') 
plt.title("Sentiment Confidence Score Distribution")
plt.xlabel("Confidence Score")
plt.savefig("Confidence_Score_Distribution.png")


plt.figure(figsize=(6,4))
# Using the new column name here
sns.countplot(data=df, x="TextBlob_Sentiment_Label", palette="Set2") 
plt.title("Sentiment Distribution of Today's Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("Sentiment_Distribution.png")


# FINAL EXPORT
df.to_csv("Client_Review_Sentiment_Report.csv", index=False)
print("✅ Exported: Client_Review_Sentiment_Report.csv")