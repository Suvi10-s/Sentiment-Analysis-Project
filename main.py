from fastapi import FastAPI
from schemas import Input
import joblib
import re
import emoji
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer

model_nb = joblib.load("sentiment analysis nb.pkl")
model_lo = joblib.load("sentiment analysis logistic.pkl")
app = FastAPI()


def emoji_to_sentiment(text):
    if not type(text)==str:
        return text
    text = emoji.demojize(text)
    happy_words = [
    "smi","grin","laugh","joy","happy","love","heart",
    "kiss","hug","blush","relief","satisfy",
    "star","sparkle","fire","glow",
    "cool","sunglass",
    "thumb","clap","ok","victory",
    "party","confetti","tada",
    "beam","cheer","excite",
    "amaze","fantastic","good",
    "hundred","muscle","rocket","rainbow"
]
    sad_words = [
    "sad","cry","sob","angry","rage",
    "frown","disappoint","break","pain","hurt",
    "fear","scare","terror","scream",
    "weary","tire","sleep","pensive","confuse",
    "grim","awkward","frustrate","annoy",
    "vomit","sick","nausea",
    "anguish","depress","miserable","upset",
    "bad","worst","steam","without","unamused",
]
    neutral_words = [
    "ok","okay","fine","normal","average",
    "neutral","meh",
    "think","consider",
    "shrug","unsure","indifferent","with",
    "blank","calm","straight"
]
    emoji_word = re.findall(r":\S+:",text)
    for emojis in emoji_word:
        exact_words = re.sub(r"[:_]+"," ",emojis)
        for word in happy_words:
            if word in exact_words:
                text = text.replace(emojis," happy ")
                break
        for word in sad_words:
            if word in exact_words:
                text = text.replace(emojis," sad ")
                break
        for word in neutral_words:
            if word in exact_words:
                text = text.replace(emojis," neutral ")
                break
    return text


def preprocess(text): 
    text = text.lower()    
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)    
    text = re.sub(r"\S+@\S+", " ", text)    
    text = re.sub(r"@\S+", " ", text)    
    text = re.sub(r"#", " ", text)
    text = emoji_to_sentiment(text)
    text = contractions.fix(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
vector =joblib.load("vectorizer.pkl")


@app.post("/naive",tags=["Naive Bayes"])
def predict_nb(data:Input):
    text = preprocess(data.text)
    vectorized = vector.transform([text])
    prediction = model_nb.predict(vectorized)
    return {"prediction::": int(prediction[0])}


@app.post("/logistic",tags=["Logistic regression"])
def predict_lo(data:Input):
    text = preprocess(data.text)
    vectorized = vector.transform([text])
    prediction =model_lo.predict(vectorized)
    return {"prediction::": int(prediction[0])}
