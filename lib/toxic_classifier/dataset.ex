defmodule ToxicClassifier.Dataset do
  @moduledoc """
  Load labelled text into `{text, :toxic | :clean}` tuples.

  Supports two shapes:

    * the Jigsaw Toxic Comment Classification `train.csv` from Kaggle
      (columns: `id, comment_text, toxic, severe_toxic, obscene, threat,
      insult, identity_hate`), where a row is `:toxic` if ANY of the six label
      columns is `1`; and

    * a simple 2-column `text,label` CSV (the bundled sample dataset).
  """

  # NimbleCSV RFC-4180 parser handles Jigsaw's quoted, multi-line comments.
  NimbleCSV.define(ToxicClassifier.CSV, separator: ",", escape: "\"")

  @jigsaw_label_cols ~w(toxic severe_toxic obscene threat insult identity_hate)

  @type example :: {String.t(), :toxic | :clean}

  @doc "Path to the bundled sample dataset (used when no dataset is passed)."
  @spec sample_path() :: String.t()
  def sample_path, do: Path.join(:code.priv_dir(:toxic_classifier), "data/sample_toxic.csv")

  @doc """
  Resolve a dataset path from CLI args: the first argument, or the bundled
  sample when none is given.
  """
  @spec path_from_args([String.t()]) :: String.t()
  def path_from_args([path | _]), do: path
  def path_from_args([]), do: sample_path()

  @doc """
  Load examples from `path`, auto-detecting the Jigsaw vs. simple format from
  the header row.
  """
  @spec load(String.t()) :: [example()]
  def load(path) do
    [header | data] =
      path
      |> File.read!()
      |> ToxicClassifier.CSV.parse_string(skip_headers: false)

    cond do
      "comment_text" in header -> load_jigsaw(header, data)
      true -> load_simple(header, data)
    end
  end

  @doc """
  Shuffle and split examples into `{train, test}` by `ratio` (default 0.8).
  """
  @spec split([example()], float(), keyword()) :: {[example()], [example()]}
  def split(examples, ratio \\ 0.8, opts \\ []) do
    case Keyword.get(opts, :seed) do
      nil -> :ok
      seed -> :rand.seed(:exsss, {seed, seed, seed})
    end

    shuffled = Enum.shuffle(examples)
    cut = round(length(shuffled) * ratio)
    Enum.split(shuffled, cut)
  end

  @doc """
  Undersample the majority class(es) down to `ratio` times the minority size.
  """
  @spec balance([example()], number()) :: [example()]
  def balance(examples, ratio) do
    by_class = Enum.group_by(examples, fn {_text, label} -> label end)
    min_size = by_class |> Map.values() |> Enum.map(&length/1) |> Enum.min()
    cap = round(min_size * ratio)

    by_class
    |> Enum.flat_map(fn {_label, items} -> items |> Enum.shuffle() |> Enum.take(cap) end)
    |> Enum.shuffle()
  end

  defp load_jigsaw(header, data) do
    text_idx = Enum.find_index(header, &(&1 == "comment_text"))
    label_idxs = Enum.map(@jigsaw_label_cols, &Enum.find_index(header, fn c -> c == &1 end))

    data
    |> Enum.map(fn row ->
      text = Enum.at(row, text_idx, "")
      toxic? = Enum.any?(label_idxs, fn idx -> idx && Enum.at(row, idx) == "1" end)
      {text, if(toxic?, do: :toxic, else: :clean)}
    end)
    |> Enum.reject(fn {text, _} -> is_nil(text) or text == "" end)
  end

  defp load_simple(header, data) do
    text_idx = Enum.find_index(header, &(&1 in ["text", "comment", "comment_text"])) || 0
    label_idx = Enum.find_index(header, &(&1 in ["label", "class", "toxic"])) || 1

    Enum.map(data, fn row ->
      text = Enum.at(row, text_idx, "")
      {text, normalize_label(Enum.at(row, label_idx))}
    end)
  end

  defp normalize_label(v) when v in ["toxic", "1", "spam", "true"], do: :toxic
  defp normalize_label(_), do: :clean
end
