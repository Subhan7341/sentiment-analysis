# SentimentIQ — Text Emotion Analyzer

A full-stack NLP web application that analyzes the emotional tone of any text using an **ensemble of two machine learning models** — TextBlob (Naive Bayes) and VADER — for higher accuracy than either model alone.

**Built with:** Python · Flask · NLTK · TextBlob · HTML · CSS · JavaScript

---

## What it does

- Classifies text as **Positive**, **Negative**, or **Neutral**
- Shows a **confidence score** and **average sentiment score**
- Displays a full **VADER breakdown** (% positive / neutral / negative)
- Shows **TextBlob polarity** (−1 to +1) and **subjectivity** (fact vs opinion)
- Highlights the most **emotionally charged words** in the text with individual scores
- Includes **6 example sentences** to try instantly

---

## Tech Stack

| Layer     | Technology |
|-----------|------------|
| Backend   | Python, Flask |
| NLP       | NLTK, TextBlob, VADER |
| Frontend  | HTML, CSS, JavaScript (no frameworks) |
| API       | REST (JSON) |

---

## How it works (the ML pipeline)

```
User types text
      ↓
Preprocessing (clean text, strip whitespace)
      ↓
  ┌─────────────┐     ┌──────────────────┐
  │  TextBlob   │     │      VADER        │
  │ Naive Bayes │     │  Rule-based NLP   │
  │ polarity    │     │  compound score   │
  └──────┬──────┘     └────────┬─────────┘
         └─────────┬───────────┘
                   ↓
          Average both scores
          (Ensemble Method)
                   ↓
         Final label + confidence
```

**Why two models?** TextBlob is better for formal text (trained on movie reviews). VADER is better for short/casual text (tweets, reviews). Averaging them is called an **ensemble approach** and gives more reliable results.

---

## Project Structure

```
sentiment-analysis/
├── app.py              # Flask backend + NLP pipeline
├── requirements.txt
├── README.md
└── templates/
    └── index.html      # Full UI (HTML + CSS + JS inline)
```

---

## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/Subhan7341/sentiment-analysis.git
cd sentiment-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Author

**Muhammad Subhan Ahmad**
GitHub: [github.com/Subhan7341](https://github.com/Subhan7341)
