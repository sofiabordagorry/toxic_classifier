defmodule PhishingClassifier.ModelBundle do
  @moduledoc """
  Trains the three classifiers into a *serialisable* bundle and rebuilds the
  per-model probability closures the LiveView renders.

  The models themselves are plain data (maps, MapSets, floats), so a bundle can
  be written to disk with `save/1` and read back with `load/0` — which is how a
  trained model survives a server restart. The rendering closures are rebuilt
  from that data on load (functions can't be serialised).
  """

  alias PhishingClassifier.{NaiveBayes, ComplementNB, LogisticRegression}

  @type bundle :: %{
          nb: NaiveBayes.t(),
          cnb: ComplementNB.t(),
          lr: LogisticRegression.t(),
          summary: String.t()
        }

  @doc "Train nb / cnb / lr from `{text, label}` examples and package them."
  @spec build([{String.t(), atom()}]) :: bundle()
  def build(examples) do
    nb = NaiveBayes.train(examples, balance_ratio: 1.0, min_count: 2)
    cnb = ComplementNB.from_model(nb)

    lr =
      LogisticRegression.train(examples,
        balance_ratio: 1.0,
        min_count: 2,
        epochs: 8,
        positive: :phish,
        negative: :safe
      )

    dist = Enum.frequencies_by(examples, fn {_text, label} -> label end)

    summary =
      "Trained on #{length(examples)} examples · " <>
        "#{Map.get(dist, :phish, 0)} phishing / #{Map.get(dist, :safe, 0)} safe"

    %{nb: nb, cnb: cnb, lr: lr, summary: summary}
  end

  @doc "Turn a bundle into the `%{classifiers, summary}` map the UI renders."
  @spec to_ui(bundle()) :: %{classifiers: [{String.t(), (String.t() -> float())}], summary: String.t()}
  def to_ui(%{nb: nb, cnb: cnb, lr: lr, summary: summary}) do
    classifiers = [
      {"Naive Bayes", &Map.get(NaiveBayes.predict_proba(nb, &1), :phish, 0.0)},
      {"Complement NB", &Map.get(ComplementNB.predict_proba(cnb, &1), :phish, 0.0)},
      {"Logistic Regression", &Map.get(LogisticRegression.predict_proba(lr, &1), :phish, 0.0)}
    ]

    %{classifiers: classifiers, summary: summary}
  end

  # --- persistence ---

  @doc "Path to the on-disk model cache."
  @spec cache_path() :: String.t()
  def cache_path do
    Path.join(:code.priv_dir(:phishing_classifier), "model_cache.bin")
  end

  @doc "Serialise a bundle to disk (compressed). Returns the bundle unchanged."
  @spec save(bundle()) :: bundle()
  def save(bundle) do
    File.write!(cache_path(), :erlang.term_to_binary(bundle, [:compressed]))
    bundle
  end

  @doc "Load a persisted bundle, or `nil` if none exists / it can't be read."
  @spec load() :: bundle() | nil
  def load do
    path = cache_path()

    with true <- File.exists?(path),
         {:ok, bin} <- File.read(path),
         %{nb: _, cnb: _, lr: _} = bundle <- decode(bin) do
      bundle
    else
      _ -> nil
    end
  end

  @doc "Delete the on-disk cache, if any."
  @spec delete() :: :ok
  def delete do
    _ = File.rm(cache_path())
    :ok
  end

  # The cache is written by this app into its own priv dir (trusted input), so
  # we decode without `:safe` — the struct atoms it contains aren't all in the
  # atom table on a fresh boot, which `:safe` would reject. A corrupt file just
  # raises and we fall back to "untrained".
  defp decode(bin) do
    :erlang.binary_to_term(bin)
  rescue
    _ -> nil
  end
end
