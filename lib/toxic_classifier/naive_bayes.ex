defmodule ToxicClassifier.NaiveBayes do
  @moduledoc """
  A multinomial Naive Bayes text classifier.

  For a document with tokens `w1..wn` and class `c`, Naive Bayes picks the class
  maximising:

      log P(c) + Σ log P(wi | c)

  where, with add-`alpha` (Laplace) smoothing over a vocabulary of size `V`:

      P(wi | c) = (count(wi, c) + alpha) / (total_tokens(c) + alpha * V)

  Working in log-space turns the product into a sum and avoids floating-point
  underflow on long documents.
  """

  alias ToxicClassifier.Tokenizer

  @enforce_keys [:classes, :class_docs, :class_tokens, :word_counts, :vocab_size, :alpha]
  defstruct [:classes, :class_docs, :class_tokens, :word_counts, :vocab_size, :alpha]

  @type label :: atom()
  @type example :: {String.t(), label()}

  @type t :: %__MODULE__{
          classes: [label()],
          class_docs: %{label() => non_neg_integer()},
          class_tokens: %{label() => non_neg_integer()},
          word_counts: %{label() => %{String.t() => non_neg_integer()}},
          vocab_size: non_neg_integer(),
          alpha: float()
        }

  @doc """
  Train a model from `{text, label}` examples.

  ## Options

    * `:alpha` - Laplace smoothing strength (default `1.0`).
    * `:balance` - when `true`, undersample the majority class to match the
      minority (equivalent to `balance_ratio: 1.0`). Default `false`.
    * `:balance_ratio` - keep the majority class at this multiple of the
      minority size (e.g. `2.0` -> a gentler 2:1 balance). Overrides `:balance`
      when set. `nil` (default) means no balancing.
    * `:min_count` - drop features whose total count across classes is below
      this value (default `1`, i.e. keep everything). Pruning rare features
      (typos, names) removes spurious signals and reduces false positives.
  """
  @spec train([example()], keyword()) :: t()
  def train(examples, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, 1.0)
    min_count = Keyword.get(opts, :min_count, 1)

    ratio =
      cond do
        Keyword.has_key?(opts, :balance_ratio) -> Keyword.get(opts, :balance_ratio)
        Keyword.get(opts, :balance, false) -> 1.0
        true -> nil
      end

    examples = if ratio, do: ToxicClassifier.Dataset.balance(examples, ratio), else: examples

    # 1. Count documents and per-class word frequencies.
    {class_docs, raw_counts} =
      Enum.reduce(examples, {%{}, %{}}, fn {text, label}, {docs, counts} ->
        tokens = Tokenizer.tokenize(text)
        {Map.update(docs, label, 1, &(&1 + 1)), add_word_counts(counts, label, tokens)}
      end)

    # 2. Prune rare features, then derive the vocabulary and per-class totals
    #    from what survived (so smoothing denominators stay consistent).
    word_counts = prune(raw_counts, min_count)
    vocab = vocab_of(word_counts)

    class_tokens =
      Map.new(word_counts, fn {class, m} -> {class, m |> Map.values() |> Enum.sum()} end)

    %__MODULE__{
      classes: Map.keys(class_docs),
      class_docs: class_docs,
      class_tokens: class_tokens,
      word_counts: word_counts,
      vocab_size: MapSet.size(vocab),
      alpha: alpha
    }
  end

  @doc "Predict the single most likely label for `text`."
  @spec predict(t(), String.t()) :: label()
  def predict(model, text) do
    model
    |> scores(text)
    |> Enum.max_by(fn {_label, score} -> score end)
    |> elem(0)
  end

  @doc """
  Predict a normalised probability per class as a map, e.g.
  `%{toxic: 0.97, clean: 0.03}`.

  Uses the log-sum-exp trick to convert log-scores back to probabilities
  stably.
  """
  @spec predict_proba(t(), String.t()) :: %{label() => float()}
  def predict_proba(model, text) do
    log_scores = scores(model, text)
    max_log = log_scores |> Enum.map(&elem(&1, 1)) |> Enum.max()

    exps = Enum.map(log_scores, fn {label, s} -> {label, :math.exp(s - max_log)} end)
    total = exps |> Enum.map(&elem(&1, 1)) |> Enum.sum()

    Map.new(exps, fn {label, e} -> {label, e / total} end)
  end

  @doc """
  Binary decision with an adjustable threshold on the `positive` class
  probability (default `:toxic`, threshold `0.5`).

  Raising the threshold trades recall for precision — fewer false positives,
  which is usually what you want in a live demo.
  """
  @spec classify(t(), String.t(), keyword()) :: label()
  def classify(model, text, opts \\ []) do
    positive = Keyword.get(opts, :positive, :toxic)
    negative = Keyword.get(opts, :negative, :clean)
    threshold = Keyword.get(opts, :threshold, 0.5)

    proba = predict_proba(model, text)
    if Map.get(proba, positive, 0.0) >= threshold, do: positive, else: negative
  end

  # Log-score per class: log P(c) + Σ log P(wi | c)
  defp scores(model, text) do
    tokens = Tokenizer.tokenize(text)
    total_docs = model.class_docs |> Map.values() |> Enum.sum()
    denom_base = model.alpha * model.vocab_size

    Enum.map(model.classes, fn class ->
      log_prior = :math.log(model.class_docs[class] / total_docs)
      class_total = model.class_tokens[class]
      counts = Map.get(model.word_counts, class, %{})

      log_likelihood =
        Enum.reduce(tokens, 0.0, fn token, sum ->
          count = Map.get(counts, token, 0)
          sum + :math.log((count + model.alpha) / (class_total + denom_base))
        end)

      {class, log_prior + log_likelihood}
    end)
  end

  defp add_word_counts(word_counts, label, tokens) do
    class_map = Map.get(word_counts, label, %{})
    updated = Enum.reduce(tokens, class_map, fn t, m -> Map.update(m, t, 1, &(&1 + 1)) end)
    Map.put(word_counts, label, updated)
  end

  # Drop any feature whose summed count across all classes is below min_count.
  defp prune(word_counts, min_count) when min_count <= 1, do: word_counts

  defp prune(word_counts, min_count) do
    totals =
      Enum.reduce(word_counts, %{}, fn {_class, m}, acc ->
        Enum.reduce(m, acc, fn {w, c}, a -> Map.update(a, w, c, &(&1 + c)) end)
      end)

    keep = for {w, total} <- totals, total >= min_count, into: MapSet.new(), do: w

    Map.new(word_counts, fn {class, m} ->
      {class, Map.take(m, MapSet.to_list(keep))}
    end)
  end

  defp vocab_of(word_counts) do
    Enum.reduce(word_counts, MapSet.new(), fn {_class, m}, acc ->
      Enum.reduce(Map.keys(m), acc, &MapSet.put(&2, &1))
    end)
  end
end
