# ToxicClassifier

A little project to try out toxic-comment classification in Elixir,
with three text classifiers.

## The three classifiers

| Module | Idea | Kind |
|---|---|---|
| `ToxicClassifier.NaiveBayes` | Multinomial Naive Bayes: count words per class, classify by summed log-probabilities. | generative |
| `ToxicClassifier.ComplementNB` | Complement NB: estimate each class from its *complement*, better on imbalanced data. | generative |
| `ToxicClassifier.LogisticRegression` | Logistic regression via stochastic gradient descent; learns a weight per feature. | discriminative |

## Web UI

```bash
mix phx.server   # then open http://localhost:4000
```

Upload a labelled CSV, it trains the three models live, and typing a phrase
shows a toxicity meter per model. No database — models live in memory per session.

## Running

Pass a dataset path, or omit it to use the bundled sample.

```bash
mix run scripts/compare.exs path/to/train.csv   # compare all three
mix run scripts/compare.exs                      # bundled sample
mix run scripts/train.exs path/to/train.csv      # single NB model + predictions
```

Dataset I used to test: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
