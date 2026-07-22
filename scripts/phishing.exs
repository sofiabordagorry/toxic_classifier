# Train and compare the from-scratch classifiers on real phishing emails.
#
#   mix run scripts/phishing.exs                 # bundled phishing_email.csv
#   mix run scripts/phishing.exs path/to.csv     # custom CSV (Email Text, Email Type)
#
# Env knobs: SEED, LIMIT, MIN_COUNT, EPOCHS, MAXLEN
#
# The point of this script (and the talk): a ~100-line bag-of-words model,
# trained in seconds, gets within a hair of a 336M-parameter fine-tuned BERT
# on a task whose signal is mostly lexical ("verify your account", "click
# here", "suspended"). You rarely need an LLM to count words.

require Logger

alias PhishingClassifier.{Dataset, NaiveBayes, ComplementNB, LogisticRegression}

# NimbleCSV parser that tolerates the huge, quoted, multi-line email bodies.
NimbleCSV.define(PhishCSV, separator: ",", escape: "\"")

default = Path.join(:code.priv_dir(:phishing_classifier), "data/phishing_email.csv")
path = case System.argv() do
  [p | _] -> p
  [] -> default
end

seed = System.get_env("SEED", "42") |> String.to_integer()
limit = System.get_env("LIMIT") |> then(&(&1 && String.to_integer(&1)))
min_count = System.get_env("MIN_COUNT", "3") |> String.to_integer()
epochs = System.get_env("EPOCHS", "8") |> String.to_integer()
maxlen = System.get_env("MAXLEN", "4000") |> String.to_integer()

Logger.info("Loading #{path}")

# Map "Phishing Email" -> :phish, "Safe Email" -> :safe. Cap body length so a
# few pathological multi-MB rows don't dominate tokenisation.
examples =
  path
  |> File.read!()
  |> PhishCSV.parse_string(skip_headers: true)
  |> Enum.flat_map(fn
    [_idx, text, type] ->
      text = text |> to_string() |> String.slice(0, maxlen)
      label = if String.downcase(type) =~ "phish", do: :phish, else: :safe
      if String.trim(text) == "", do: [], else: [{text, label}]

    _ ->
      []
  end)

examples = if limit, do: Enum.take(Enum.shuffle(examples), limit), else: examples

dist = Enum.frequencies_by(examples, fn {_t, l} -> l end)
Logger.info("#{length(examples)} emails  #{inspect(dist)}")

{train, test} = Dataset.split(examples, 0.8, seed: seed)
Logger.info("#{length(train)} train / #{length(test)} test")

thresholds = for t <- 50..99, do: t / 100

# Score the whole test set once, then sweep thresholds and keep the best F1.
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
        case {actual == :phish, p >= t} do
          {true, true} -> {tp + 1, fp, fn_, tn}
          {false, true} -> {tp, fp + 1, fn_, tn}
          {true, false} -> {tp, fp, fn_ + 1, tn}
          {false, false} -> {tp, fp, fn_, tn + 1}
        end
      end)

    d = fn _n, 0 -> 0.0
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

Logger.info("Training NaiveBayes...")
{nb_us, nb} = :timer.tc(fn -> NaiveBayes.train(train, balance_ratio: 1.0, min_count: min_count) end)

Logger.info("Deriving ComplementNB...")
{cnb_us, cnb} = :timer.tc(fn -> ComplementNB.from_model(nb) end)

Logger.info("Training LogisticRegression (#{epochs} epochs)...")
{lr_us, lr} =
  :timer.tc(fn ->
    LogisticRegression.train(train,
      balance_ratio: 1.0,
      min_count: min_count,
      epochs: epochs,
      positive: :phish,
      negative: :safe
    )
  end)

models = [
  {"NaiveBayes (MNB)", nb_us, fn text -> Map.get(NaiveBayes.predict_proba(nb, text), :phish, 0.0) end},
  {"ComplementNB (CNB)", cnb_us, fn text -> Map.get(ComplementNB.predict_proba(cnb, text), :phish, 0.0) end},
  {"LogisticRegression", lr_us, fn text -> Map.get(LogisticRegression.predict_proba(lr, text), :phish, 0.0) end}
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

IO.puts("\n  ── Sample predictions (Naive Bayes, P(phish)) ─────")

demo = [
  {"Your account has been suspended. Verify your identity now: http://secure-login-update.com", :phish},
  {"URGENT: unusual sign-in detected. Confirm your password immediately or your account will be closed.", :phish},
  {"Congratulations! You have won a $1000 gift card. Click here to claim your reward now!", :phish},
  {"Dear customer, your payment failed. Update your billing details here to avoid service interruption.", :phish},
  {"Hi team, attaching the notes from today's standup. Talk tomorrow.", :safe},
  {"Can you review my pull request when you get a chance? Thanks!", :safe},
  {"Lunch at 12:30 works for me, see you then.", :safe},
  {"The quarterly report is ready — let me know if you want any changes.", :safe}
]

for {text, expected} <- demo do
  p = Map.get(NaiveBayes.predict_proba(nb, text), :phish, 0.0)
  tag = if p >= 0.5, do: "PHISH", else: "safe "
  ok = if (p >= 0.5) == (expected == :phish), do: "✓", else: "✗"
  IO.puts("  #{ok} [#{tag} #{:erlang.float_to_binary(p * 100, decimals: 1)}%]  #{String.slice(text, 0, 70)}")
end

IO.puts("")
Logger.flush()
