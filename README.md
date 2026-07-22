# PhishingClassifier

A little project exploring a simple question in Elixir: **do you really need an
LLM to detect phishing?** It pits three from-scratch, pure-Elixir text
classifiers against a 336M-parameter fine-tuned BERT — on a task whose signal is
mostly lexical ("verify your account", "click here", "suspended").

## The three from-scratch classifiers

| Module | Idea | Kind |
|---|---|---|
| `PhishingClassifier.NaiveBayes` | Multinomial Naive Bayes: count words per class, classify by summed log-probabilities. | generative |
| `PhishingClassifier.ComplementNB` | Complement NB: estimate each class from its *complement*, better on imbalanced data. | generative |
| `PhishingClassifier.LogisticRegression` | Logistic regression via stochastic gradient descent; learns a weight per feature. | discriminative |

Plus `PhishingClassifier.Bert` — the heavyweight foil: [`ealvaradob/bert-finetuned-phishing`](https://huggingface.co/ealvaradob/bert-finetuned-phishing)
(bert-large, 336M params) loaded in Elixir via [Bumblebee](https://hexdocs.pm/bumblebee).

## Results (18,631 real emails, 80/20 split)

| Model | F1 | Precision | Recall | Accuracy | Train time |
|---|---|---|---|---|---|
| Naive Bayes | 96.7% | 95.0% | 98.5% | 97.4% | ~79 s |
| Complement NB | 96.9% | 95.2% | 98.7% | 97.5% | <1 s |
| Logistic Regression | 96.6% | 94.5% | 98.7% | 97.3% | ~113 s |
| **BERT (reference)** | — | — | — | **~97.2%** | pretrained |

A ~100-line word-counter, trained in seconds on a laptop, ties a 336M-parameter
transformer. That's the whole point.

## Web UI

```bash
mix phx.server   # then open http://localhost:4000
```

Upload a labelled CSV, it trains the three models live, and typing a message
shows a phishing meter per model — compared side by side with phishing-BERT.
No database — models live in memory per session.

## Running

```bash
mix run scripts/phishing.exs                     # bundled phishing_email.csv
mix run scripts/phishing.exs path/to/emails.csv  # custom CSV (Email Text, Email Type)
```

Env knobs: `SEED`, `LIMIT`, `MIN_COUNT`, `EPOCHS`, `MAXLEN`, `ENABLE_BERT=false`.

Dataset: [Phishing Email Detection](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
(18.6k English emails, phishing vs. safe).
