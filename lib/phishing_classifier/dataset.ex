defmodule PhishingClassifier.Dataset do
  @moduledoc """
  Load labelled text into `{text, :phish | :safe}` tuples.

  Supports two shapes:

    * the phishing email CSV (columns `,Email Text,Email Type`, where Email Type
      is "Phishing Email" or "Safe Email"); and

    * a simple 2-column `text,label` CSV.
  """

  # NimbleCSV RFC-4180 parser handles the quoted, multi-line email bodies.
  NimbleCSV.define(PhishingClassifier.CSV, separator: ",", escape: "\"")

  @type example :: {String.t(), :phish | :safe}

  @doc """
  Load examples from `path`, auto-detecting the phishing vs. simple format from
  the header row.
  """
  @spec load(String.t()) :: [example()]
  def load(path) do
    [header | data] =
      path
      |> File.read!()
      |> PhishingClassifier.CSV.parse_string(skip_headers: false)

    if "Email Text" in header,
      do: load_phishing(header, data),
      else: load_simple(header, data)
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

  # Phishing email CSV: columns `,Email Text,Email Type` where Email Type is
  # "Phishing Email" or "Safe Email".
  defp load_phishing(header, data) do
    text_idx = Enum.find_index(header, &(&1 == "Email Text"))
    type_idx = Enum.find_index(header, &(&1 == "Email Type"))

    data
    |> Enum.map(fn row ->
      text = Enum.at(row, text_idx, "")
      type = row |> Enum.at(type_idx, "") |> to_string() |> String.downcase()
      {text, if(type =~ "phish", do: :phish, else: :safe)}
    end)
    |> Enum.reject(fn {text, _} -> is_nil(text) or String.trim(to_string(text)) == "" end)
  end

  defp load_simple(header, data) do
    text_idx = Enum.find_index(header, &(&1 in ["text", "message", "email", "body"])) || 0

    label_idx =
      Enum.find_index(header, &(&1 in ["label", "labels", "class", "type"])) || 1

    Enum.map(data, fn row ->
      text = Enum.at(row, text_idx, "")
      {text, normalize_label(Enum.at(row, label_idx))}
    end)
  end

  @phish_labels ~w(phish phishing spam malicious fraud scam)

  defp normalize_label(v) do
    case v |> to_string() |> String.trim() |> String.downcase() do
      s when s in @phish_labels ->
        :phish

      s ->
        case Float.parse(s) do
          {n, _} when n > 0 -> :phish
          _ -> :safe
        end
    end
  end
end
