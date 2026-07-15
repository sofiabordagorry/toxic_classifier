defmodule ToxicClassifier.ComplementNB do
  @moduledoc """
  Weight-normalised Complement Naive Bayes (WCNB).

  Standard multinomial NB struggles on skewed data. Complement NB fixes this
  by estimating each class's weights from its complement (the counts in all
  other classes) and picking the class whose complement fits the document
  worst.

  This builds directly on a trained `%ToxicClassifier.NaiveBayes{}`
  (it reuses its per-class word counts), so pruning / balancing options carry over.
  """

  alias ToxicClassifier.Tokenizer

  @enforce_keys [:classes, :weights, :defaults]
  defstruct [:classes, :weights, :defaults]

  @type t :: %__MODULE__{
          classes: [atom()],
          # class => %{feature => normalised complement weight}
          weights: %{atom() => %{String.t() => float()}},
          # class => weight assigned to features unseen in the complement
          defaults: %{atom() => float()}
        }

  @doc "Build a WCNB model from a trained `%ToxicClassifier.NaiveBayes{}`."
  @spec from_model(ToxicClassifier.NaiveBayes.t()) :: t()
  def from_model(%ToxicClassifier.NaiveBayes{} = m, opts \\ []) do
    v = m.vocab_size
    alpha = m.alpha

    normalize = Keyword.get(opts, :normalize, false)

    {weights, defaults} =
      Enum.reduce(m.classes, {%{}, %{}}, fn class, {w_acc, d_acc} ->
        others = Enum.reject(m.classes, &(&1 == class))

        # Complement counts: sum this feature over every OTHER class.
        comp_counts =
          Enum.reduce(others, %{}, fn oc, acc ->
            Enum.reduce(Map.get(m.word_counts, oc, %{}), acc, fn {feat, c}, a ->
              Map.update(a, feat, c, &(&1 + c))
            end)
          end)

        comp_total = others |> Enum.map(&Map.get(m.class_tokens, &1, 0)) |> Enum.sum()

        # Raw complement log-weights: log P(feature | complement of class).
        denom = comp_total + alpha * v
        raw = Map.new(comp_counts, fn {feat, c} -> {feat, :math.log((c + alpha) / denom)} end)
        default = :math.log(alpha / denom)

        # Optional L1 normalisation over the seen features (WCNB).
        l1 =
          if normalize do
            sum = raw |> Map.values() |> Enum.reduce(0.0, fn x, s -> s + abs(x) end)
            if sum == 0.0, do: 1.0, else: sum
          else
            1.0
          end

        normed = Map.new(raw, fn {feat, x} -> {feat, x / l1} end)

        {Map.put(w_acc, class, normed), Map.put(d_acc, class, default / l1)}
      end)

    %__MODULE__{classes: m.classes, weights: weights, defaults: defaults}
  end

  @doc "Train straight from examples (convenience: builds the MNB then converts)."
  @spec train([{String.t(), atom()}], keyword()) :: t()
  def train(examples, opts \\ []) do
    examples |> ToxicClassifier.NaiveBayes.train(opts) |> from_model()
  end

  @doc """
  Per-class probability map. Lower complement score = better fit, so we softmax
  the *negated* scores.
  """
  @spec predict_proba(t(), String.t()) :: %{atom() => float()}
  def predict_proba(model, text) do
    tf = term_frequencies(text)

    # score(class) = Σ tf(t) * complement_weight(class, t) ; argmin is best.
    scores =
      Enum.map(model.classes, fn class ->
        w = model.weights[class]
        default = model.defaults[class]

        s =
          Enum.reduce(tf, 0.0, fn {feat, f}, sum ->
            sum + f * Map.get(w, feat, default)
          end)

        # Negate: higher = more likely this class, so softmax works normally.
        {class, -s}
      end)

    max = scores |> Enum.map(&elem(&1, 1)) |> Enum.max()
    exps = Enum.map(scores, fn {c, s} -> {c, :math.exp(s - max)} end)
    total = exps |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    Map.new(exps, fn {c, e} -> {c, e / total} end)
  end

  @doc "Binary decision with adjustable threshold on the positive class."
  @spec classify(t(), String.t(), keyword()) :: atom()
  def classify(model, text, opts \\ []) do
    positive = Keyword.get(opts, :positive, :toxic)
    negative = Keyword.get(opts, :negative, :clean)
    threshold = Keyword.get(opts, :threshold, 0.5)

    proba = predict_proba(model, text)
    if Map.get(proba, positive, 0.0) >= threshold, do: positive, else: negative
  end

  # Sublinear term frequency: log(1 + raw_count) per feature in the document.
  defp term_frequencies(text) do
    text
    |> Tokenizer.tokenize()
    |> Enum.frequencies()
    |> Map.new(fn {feat, count} -> {feat, :math.log(1 + count)} end)
  end
end
