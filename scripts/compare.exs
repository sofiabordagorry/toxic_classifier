# Train and compare all three classifiers on the same train/test split.
#
#   mix run scripts/compare.exs           # full dataset (slow: minutes)
#   LIMIT=40000 mix run scripts/compare.exs
#
# Env knobs: SEED, LIMIT, MIN_COUNT, EPOCHS

require Logger

alias ToxicClassifier.{Dataset, NaiveBayes, ComplementNB, LogisticRegression}

path = Dataset.path_from_args(System.argv())

seed = System.get_env("SEED", "42") |> String.to_integer()
limit = System.get_env("LIMIT") |> then(&(&1 && String.to_integer(&1)))
min_count = System.get_env("MIN_COUNT", "3") |> String.to_integer()
epochs = System.get_env("EPOCHS", "8") |> String.to_integer()

Logger.info("Loading #{path}")
examples = Dataset.load(path)
examples = if limit, do: Enum.take(Enum.shuffle(examples), limit), else: examples
{train, test} = Dataset.split(examples, 0.8, seed: seed)
Logger.info("#{length(train)} train / #{length(test)} test")

thresholds = for t <- 50..99, do: t / 100

# Score the whole test set once, then sweep thresholds cheaply and keep best F1.
best_metrics = fn proba_fn ->
  scored =
    test
    |> Task.async_stream(fn {text, actual} -> {actual, proba_fn.(text)} end,
      ordered: false,
      timeout: :infinity
    )
    |> Enum.map(fn {:ok, pair} -> pair end)

  Enum.map(thresholds, fn t ->
    {tp, fp, fn_, tn} =
      Enum.reduce(scored, {0, 0, 0, 0}, fn {actual, p}, {tp, fp, fn_, tn} ->
        case {actual == :toxic, p >= t} do
          {true, true} -> {tp + 1, fp, fn_, tn}
          {false, true} -> {tp, fp + 1, fn_, tn}
          {true, false} -> {tp, fp, fn_ + 1, tn}
          {false, false} -> {tp, fp, fn_, tn + 1}
        end
      end)

    d = fn _n, den when den == 0 -> 0.0
          n, den -> n / den end

    precision = d.(tp, tp + fp)
    recall = d.(tp, tp + fn_)
    %{
      t: t,
      f1: d.(2 * precision * recall, precision + recall),
      precision: precision,
      recall: recall,
      acc: d.(tp + tn, tp + fp + fn_ + tn)
    }
  end)
  |> Enum.max_by(& &1.f1)
end

# Train each model, timing it, and get its proba function.
Logger.info("Training NaiveBayes...")
{nb_us, nb} = :timer.tc(fn -> NaiveBayes.train(train, balance_ratio: 1.0, min_count: min_count) end)

Logger.info("Deriving ComplementNB...")
{cnb_us, cnb} = :timer.tc(fn -> ComplementNB.from_model(nb) end)

Logger.info("Training LogisticRegression (#{epochs} epochs)...")
{lr_us, lr} =
  :timer.tc(fn ->
    LogisticRegression.train(train, balance_ratio: 1.0, min_count: min_count, epochs: epochs)
  end)

models = [
  {"NaiveBayes (MNB)", nb_us, fn text -> Map.get(NaiveBayes.predict_proba(nb, text), :toxic, 0.0) end},
  {"ComplementNB (CNB)", cnb_us, fn text -> Map.get(ComplementNB.predict_proba(cnb, text), :toxic, 0.0) end},
  {"LogisticRegression", lr_us, fn text -> Map.get(LogisticRegression.predict_proba(lr, text), :toxic, 0.0) end}
]

IO.puts("\n  model                  train     F1     precision  recall   accuracy  @thr")
IO.puts("  ────────────────────────────────────────────────────────────────────────")

for {name, train_us, proba_fn} <- models do
  Logger.info("Scoring #{name}...")
  m = best_metrics.(proba_fn)
  pct = fn x -> String.pad_leading(:erlang.float_to_binary(x * 100, decimals: 1), 6) end

  IO.puts(
    "  #{String.pad_trailing(name, 20)}  #{String.pad_leading(:erlang.float_to_binary(train_us / 1_000, decimals: 0), 5)}ms  " <>
      "#{pct.(m.f1)}%  #{pct.(m.precision)}%   #{pct.(m.recall)}%  #{pct.(m.acc)}%  #{:erlang.float_to_binary(m.t, decimals: 2)}"
  )
end

IO.puts("")
Logger.flush()
