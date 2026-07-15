defmodule ToxicClassifier.Evaluation do
  @moduledoc """
  Metrics for a binary classifier: confusion matrix, accuracy, and per-positive
  precision / recall / F1.
  """

  require Logger

  @type example :: {String.t(), atom()}

  @doc """
  Evaluate `model` against labelled `examples` using `classify_fn`, a function
  that maps text -> predicted label. Returns a metrics map.
  """
  @spec evaluate(module_or_any :: any(), [example()], (any(), String.t() -> atom()), atom()) ::
          map()
  def evaluate(model, examples, classify_fn, positive \\ :toxic) do
    total = length(examples)

    counts =
      examples
      |> Task.async_stream(
        fn {text, actual} -> {actual, classify_fn.(model, text)} end,
        ordered: false,
        timeout: :infinity
      )
      |> Enum.reduce({%{tp: 0, fp: 0, tn: 0, fn: 0}, 0}, fn {:ok, {actual, predicted}},
                                                            {acc, done} ->
        done = done + 1
        if rem(done, 2000) == 0, do: Logger.info("evaluated #{done}/#{total}")
        {bump(acc, actual, predicted, positive), done}
      end)
      |> elem(0)

    total = counts.tp + counts.tn + counts.fp + counts.fn
    precision = safe_div(counts.tp, counts.tp + counts.fp)
    recall = safe_div(counts.tp, counts.tp + counts.fn)

    %{
      total: total,
      accuracy: safe_div(counts.tp + counts.tn, total),
      precision: precision,
      recall: recall,
      f1: safe_div(2 * precision * recall, precision + recall),
      confusion: counts
    }
  end

  @doc "Pretty-print a metrics map to stdout."
  @spec report(map()) :: :ok
  def report(m) do
    c = m.confusion

    IO.puts("""

    ── Evaluation (#{m.total} test docs) ──────────────
      Accuracy : #{pct(m.accuracy)}
      Precision: #{pct(m.precision)}   (of flagged-toxic, how many really were)
      Recall   : #{pct(m.recall)}   (of truly-toxic, how many we caught)
      F1       : #{pct(m.f1)}

      Confusion matrix
                     pred toxic   pred clean
        act toxic        #{pad(c.tp)}        #{pad(c.fn)}
        act clean        #{pad(c.fp)}        #{pad(c.tn)}
    ───────────────────────────────────────────────────
    """)
  end

  # --- helpers ---

  defp bump(acc, actual, predicted, pos) do
    key =
      case {actual == pos, predicted == pos} do
        {true, true} -> :tp
        {false, true} -> :fp
        {false, false} -> :tn
        {true, false} -> :fn
      end

    Map.update!(acc, key, &(&1 + 1))
  end

  defp safe_div(_num, 0), do: 0.0
  defp safe_div(num, den), do: num / den

  defp pct(x), do: :erlang.float_to_binary(x * 100, decimals: 2) <> "%"
  defp pad(n), do: String.pad_leading(Integer.to_string(n), 6)
end
