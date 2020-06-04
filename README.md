# Kaggle tweet sentiment extraction
https://www.kaggle.com/c/tweet-sentiment-extraction/overview

The objective in this competition is to construct a model that can look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.

## Dataset
Each row contains the text of a tweet and a sentiment label. In the training set you are provided with a word or phrase drawn from the tweet (selected_text) that encapsulates the provided sentiment.
Data:
* Text: string
* Sentiment: {"positive", "negative", "neutral"}

Prediction:
* Selected text: string

You're attempting to predict the word or phrase from the tweet that exemplifies the provided sentiment. The word or phrase should include all characters within that span (i.e. including commas, spaces, etc.).

## Metrics
The metric in this competition is the [word-level Jaccard score](https://en.wikipedia.org/wiki/Jaccard_index). 

```python
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

## Baseline
Selected text is simply a copy of initial text.
Gives a score of 0.594 on submission.

