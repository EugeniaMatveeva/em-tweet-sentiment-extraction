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

## Attempted approaches
*EugeniaMatveeva (@eugenie_mat)*
1. RNN with pretrained GloVe embeddings (RNN start-end predict model):

-Embedding Layer (64x300)

-Bidirectional GRU (300x2\*300)

-ReLU

-Dropout (0.3)

-Linear Layer1 + LogSoftmax (2\*300x1) (start position)

-Linear Layer2 + LogSoftmax (2\*300x1) (end position)


Text is preprocessed and split into words and position of start and end words are predicted. Sum log probabilities of start and end words and use Negative Log Likelihood loss.

Score on validation set is  0.721 with fine tuned parameters (learning rate, hidden size, dropout, epochs).
Submission score is 0.552.

I also tried training separate models for positive, negative and neutral sentiments but it didn't improve the score.

2.  RNN with pretrained GloVe embeddings and attention (RNN with attention):
Tried to add attention to the previous RNN model using [Luong Attention](https://arxiv.org/pdf/1508.04025.pdf). 
The idea is to add attention as it is used question answering tasks: tweet text corresponds to text, sentiment corresponds to the question and answer should be selected part of the text.
For attention mechanism sentiment embedding is used as a query, RNN outputs for each text word are keys.

Attention didn't improve the model, the loss falls only for 2-3 of epochs and the model overfits. Parameter tuning didn't help and score was very low on validation set - around 0.41.
My explanation is that one word sentiment may carry too little context to make attention work well here. For good performance all words in selected text have to be distinctly "positively" or "negatively" colored.

*ViktorPokazanyev (@wildyagup)*
1. RoBERTa baseline:

- Pretrained roberta-base model from HuggingFace (https://huggingface.co/roberta-base)

- Sentiment prepended to input text as separate sentence

- Averaged 3 last hiddden layers for better performance

- Two linear layers for start/end positions prediction

- Masked output for CrossEntropyLoss calculation and prediciton (all special tokens and sentiment)

- Prediction is obtained via an algorithm described in BIDAF paper (maximization of p_start * p_end)

- AdamW + linear learning rate scheduler

Best leaderboard score is 0.701 (~0.72-0.74 on validation).

## Future plans:
*EugeniaMatveeva (@eugenie_mat)*

3. BERT model
Try to use pretrained BERT model. Hopefully this will give better scores as it is more efficient, pretrained model. Also it gives better tokenisation and works better with unknown words which is a big problem for twitter nonstrict vocabulary.

4. Many selected texts are not strictly split, they include punctuation or parts of words. So adding some character level models as CNN may help, maybe in ensemble with another model that works best.

*ViktorPokazanyev (@wildyagup)*

2. RoBERTa + RNN
An attempt to imitate BIDAF architecture's modelling and output layers while replacing attention flow with transformer. 
The main challenge is to prevent RNN overfitting.

3. Miscellaneous improvements
* experiment with output masking / padding / batching
* experiment with tokenization (word parts / ngrams)