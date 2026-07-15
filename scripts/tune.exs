require Logger

alias ToxicClassifier.Dataset

path = Dataset.path_from_args(System.argv())

alpha = System.get_env("ALPHA", "1.0") |> String.to_float()
seed = System.get_env("SEED", "42") |> String.to_integer()
limit = System.get_env("LIMIT") |> then(&(&1 && String.to_integer(&1)))
min_count = System.get_env("MIN_COUNT", "1") |> String.to_integer()
# RATIO: majority:minority balance (e.g. 1.0 = 50/50, 3.0 = gentler, "off" = none)
ratio =
  case System.get_env("RATIO", "1.0") do
    "off" -> nil
    v -> String.to_float(v)
  end

Logger.info("Loading #{path}")
examples = Dataset.load(path)
examples = if limit, do: Enum.take(Enum.shuffle(examples), limit), else: examples

mode = System.get_env("MODE", "mnb")

{train, test} = Dataset.split(examples, 0.8, seed: seed)
Logger.info("#{length(train)} train / #{length(test)} test — training (mode=#{mode}, ratio=#{inspect(ratio)}, min_count=#{min_count})...")
mnb = ToxicClassifier.NaiveBayes.train(train, alpha: alpha, balance_ratio: ratio, min_count: min_count)

proba_fn =
  case mode do
    "cnb" ->
      cnb = ToxicClassifier.ComplementNB.from_model(mnb)
      fn text -> Map.get(ToxicClassifier.ComplementNB.predict_proba(cnb, text), :toxic, 0.0) end

    _ ->
      fn text -> Map.get(ToxicClassifier.NaiveBayes.predict_proba(mnb, text), :toxic, 0.0) end
  end

Logger.info("Trained (vocab #{mnb.vocab_size}). Scoring #{length(test)} test docs once...")

# Expensive step, done ONCE: P(toxic) for every test doc, in parallel.
scored =
  test
  |> Task.async_stream(
    fn {text, actual} -> {actual, proba_fn.(text)} end,
    ordered: false,
    timeout: :infinity
  )
  |> Enum.map(fn {:ok, pair} -> pair end)

Logger.info("Scored. Sweeping thresholds (cheap)...")

thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

metrics_at = fn t ->
  {tp, fp, fn_, tn} =
    Enum.reduce(scored, {0, 0, 0, 0}, fn {actual, p}, {tp, fp, fn_, tn} ->
      case {actual == :toxic, p >= t} do
        {true, true} -> {tp + 1, fp, fn_, tn}
        {false, true} -> {tp, fp + 1, fn_, tn}
        {true, false} -> {tp, fp, fn_ + 1, tn}
        {false, false} -> {tp, fp, fn_, tn + 1}
      end
    end)

  div0 = fn _n, 0 -> 0.0
             n, d -> n / d end

  precision = div0.(tp, tp + fp)
  recall = div0.(tp, tp + fn_)
  f1 = div0.(2 * precision * recall, precision + recall)
  acc = div0.(tp + tn, tp + fp + fn_ + tn)
  %{t: t, acc: acc, precision: precision, recall: recall, f1: f1, fp: fp, fn: fn_}
end

rows = Enum.map(thresholds, metrics_at)
best = Enum.max_by(rows, & &1.f1)

pct = fn x -> :erlang.float_to_binary(x * 100, decimals: 1) |> String.pad_leading(6) end

IO.puts("\n  thr   accuracy  precision   recall     F1     false+  false-")
IO.puts("  ─────────────────────────────────────────────────────────────")

for r <- rows do
  mark = if r == best, do: "  <- best F1", else: ""
  IO.puts(
    "  #{:erlang.float_to_binary(r.t, decimals: 2)}  #{pct.(r.acc)}%  #{pct.(r.precision)}%  " <>
      "#{pct.(r.recall)}%  #{pct.(r.f1)}%  #{String.pad_leading("#{r.fp}", 6)}  " <>
      "#{String.pad_leading("#{r.fn}", 6)}#{mark}"
  )
end

IO.puts("\n  Best F1 = #{pct.(best.f1)}% at threshold #{:erlang.float_to_binary(best.t, decimals: 2)}\n")

Logger.flush()
