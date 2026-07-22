defmodule PhishingClassifier.Tokenizer do
  @moduledoc """
  Turns raw text into a list of feature tokens for the classifiers.

    * lowercases and normalises common leetspeak obfuscation (`4ccount`, `p@ssword`)
    * collapses character floods (`freeeee` -> `free`, `!!!!!` -> `!!`)
    * keeps emphasis *signals* that plain tokenisers throw away:
        - `__allcaps__` when a word is SHOUTING (phishing loves "URGENT")
        - `!` and `?` as their own tokens
    * emits both unigrams and bigrams, so "verify your" + "your account" carry
      the phrase-level context a bag-of-words would miss.
  """

  # Leetspeak / obfuscation map, applied only inside alphanumeric tokens.
  @leet %{
    "0" => "o",
    "1" => "i",
    "3" => "e",
    "4" => "a",
    "5" => "s",
    "7" => "t",
    "@" => "a",
    "$" => "s"
  }

  @doc """
  Tokenise `text` into unigrams + bigrams plus signal tokens.

  Returns a list of strings (features). Order is not meaningful to the model,
  but bigrams are built from the normalised word sequence.
  """
  @spec tokenize(String.t()) :: [String.t()]
  def tokenize(text) when is_binary(text) do
    signals = signal_tokens(text)

    words =
      text
      |> String.downcase()
      |> extract_words()
      |> Enum.map(&normalize_word/1)
      |> Enum.reject(&(&1 == ""))

    words ++ bigrams(words) ++ signals
  end

  def tokenize(_), do: []

  defp signal_tokens(text) do
    allcaps =
      text
      |> extract_words()
      |> Enum.filter(&shouting?/1)
      |> Enum.map(fn _ -> "__allcaps__" end)

    excl = if String.contains?(text, "!"), do: ["!"], else: []
    ques = if String.contains?(text, "?"), do: ["?"], else: []

    allcaps ++ excl ++ ques
  end

  # A word counts as shouting if it has >= 3 letters and they're all uppercase.
  defp shouting?(word) do
    letters = String.replace(word, ~r/[^\p{L}]/u, "")

    String.length(letters) >= 3 and String.upcase(letters) == letters and
      String.downcase(letters) != letters
  end

  # Grab word-ish chunks: letters/digits plus a few in-word symbols used for
  # obfuscation (@, $). Everything else is a separator.
  defp extract_words(text) do
    Regex.scan(~r/[\p{L}\p{N}@$]+/u, text)
    |> Enum.map(&hd/1)
  end

  defp normalize_word(word) do
    word
    |> deleet()
    |> collapse_floods()
  end

  defp deleet(word) do
    word
    |> String.graphemes()
    |> Enum.map(&Map.get(@leet, &1, &1))
    |> Enum.join()
    # after substitution, drop any leftover non-letters (e.g. stray symbols)
    |> String.replace(~r/[^\p{L}]/u, "")
  end

  # Collapse any run of 3+ identical chars down to 2 ("soooo" -> "soo").
  defp collapse_floods(word) do
    String.replace(word, ~r/(.)\1{2,}/u, "\\1\\1")
  end

  defp bigrams(words) when length(words) < 2, do: []

  defp bigrams(words) do
    words
    |> Enum.zip(tl(words))
    |> Enum.map(fn {a, b} -> a <> "_" <> b end)
  end
end
