defmodule ToxicClassifier.LogisticRegression do
  @moduledoc """
  Binary logistic regression trained from scratch with stochastic gradient
  descent.
  """

  alias ToxicClassifier.{Dataset, Tokenizer}

  @enforce_keys [:weights, :bias, :vocab, :positive, :negative]
  defstruct [:weights, :bias, :vocab, :positive, :negative]

  @type t :: %__MODULE__{
          weights: %{String.t() => float()},
          bias: float(),
          vocab: MapSet.t(),
          positive: atom(),
          negative: atom()
        }

  @doc """
  Train a logistic regression model.

  ## Options

    * `:epochs` - passes over the data (default `5`).
    * `:learning_rate` - SGD step size (default `0.1`).
    * `:l2` - L2 regularisation strength (default `0.0`).
    * `:min_count` - drop features appearing in fewer than this many documents
      (default `2`).
    * `:balance_ratio` - balance classes before training (see `Dataset.balance/2`);
      `nil` (default) trains on the data as-is.
    * `:positive` / `:negative` - class labels (default `:toxic` / `:clean`).
  """
  @spec train([{String.t(), atom()}], keyword()) :: t()
  def train(examples, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 5)
    lr = Keyword.get(opts, :learning_rate, 0.1)
    l2 = Keyword.get(opts, :l2, 0.0)
    min_count = Keyword.get(opts, :min_count, 2)
    ratio = Keyword.get(opts, :balance_ratio)
    positive = Keyword.get(opts, :positive, :toxic)
    negative = Keyword.get(opts, :negative, :clean)

    examples = if ratio, do: Dataset.balance(examples, ratio), else: examples

    # Tokenise every doc ONCE into a unique feature list + numeric label.
    docs =
      Enum.map(examples, fn {text, label} ->
        {features(text), if(label == positive, do: 1.0, else: 0.0)}
      end)

    vocab = build_vocab(docs, min_count)

    # Keep only in-vocab features per doc so the inner loop stays tight.
    docs =
      Enum.map(docs, fn {feats, y} -> {Enum.filter(feats, &MapSet.member?(vocab, &1)), y} end)

    {weights, bias} =
      Enum.reduce(1..epochs, {%{}, 0.0}, fn _epoch, state ->
        docs
        |> Enum.shuffle()
        |> Enum.reduce(state, fn {feats, y}, {w, b} -> sgd_step(w, b, feats, y, lr, l2) end)
      end)

    %__MODULE__{
      weights: weights,
      bias: bias,
      vocab: vocab,
      positive: positive,
      negative: negative
    }
  end

  @doc "Per-class probability map, e.g. `%{toxic: 0.91, clean: 0.09}`."
  @spec predict_proba(t(), String.t()) :: %{atom() => float()}
  def predict_proba(model, text) do
    p =
      text
      |> features()
      |> Enum.filter(&MapSet.member?(model.vocab, &1))
      |> score(model)
      |> sigmoid()

    %{model.positive => p, model.negative => 1.0 - p}
  end

  @doc "Most likely label (0.5 decision boundary)."
  @spec predict(t(), String.t()) :: atom()
  def predict(model, text), do: classify(model, text, threshold: 0.5)

  @doc "Binary decision with an adjustable threshold on the positive class."
  @spec classify(t(), String.t(), keyword()) :: atom()
  def classify(model, text, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.5)
    proba = predict_proba(model, text)
    if Map.get(proba, model.positive, 0.0) >= threshold, do: model.positive, else: model.negative
  end

  # One SGD update: shift the bias and every present feature's weight against
  # the gradient of log-loss. Since x[f] = 1 for present features, the gradient
  # w.r.t. weight[f] is simply the prediction error (+ L2 term).
  defp sgd_step(weights, bias, feats, y, lr, l2) do
    error = sigmoid(score(feats, %{weights: weights, bias: bias})) - y

    weights =
      Enum.reduce(feats, weights, fn f, acc ->
        wf = Map.get(acc, f, 0.0)
        Map.put(acc, f, wf - lr * (error + l2 * wf))
      end)

    {weights, bias - lr * error}
  end

  # bias + Σ weight[f] over present features. The map pattern also matches the
  # %LogisticRegression{} struct, so this serves both training and inference.
  defp score(feats, %{weights: w, bias: b}),
    do: Enum.reduce(feats, b, &(&2 + Map.get(w, &1, 0.0)))

  # Numerically stable logistic sigmoid.
  defp sigmoid(z) when z >= 0.0, do: 1.0 / (1.0 + :math.exp(-z))

  defp sigmoid(z) do
    e = :math.exp(z)
    e / (1.0 + e)
  end

  defp features(text), do: text |> Tokenizer.tokenize() |> Enum.uniq()

  # Keep features that appear in at least `min_count` documents.
  defp build_vocab(docs, min_count) do
    doc_freq =
      Enum.reduce(docs, %{}, fn {feats, _y}, acc ->
        Enum.reduce(feats, acc, fn f, a -> Map.update(a, f, 1, &(&1 + 1)) end)
      end)

    for {f, c} <- doc_freq, c >= min_count, into: MapSet.new(), do: f
  end
end
