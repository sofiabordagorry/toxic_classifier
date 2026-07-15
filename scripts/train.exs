# Train + evaluate the from-scratch Naive Bayes toxicity classifier.

require Logger

alias ToxicClassifier.{Dataset, Evaluation}

path = Dataset.path_from_args(System.argv())

alpha = System.get_env("ALPHA", "1.0") |> String.to_float()
balance = System.get_env("BALANCE", "1") == "1"
threshold = System.get_env("THRESHOLD", "0.5") |> String.to_float()
seed = System.get_env("SEED", "42") |> String.to_integer()
limit = System.get_env("LIMIT") |> then(&(&1 && String.to_integer(&1)))

Logger.info("Loading: #{path}")
{load_us, examples} = :timer.tc(fn -> Dataset.load(path) end)
Logger.info("Loaded #{length(examples)} rows in #{Float.round(load_us / 1_000, 1)} ms")

examples = if limit, do: Enum.take(Enum.shuffle(examples), limit), else: examples
dist = Enum.frequencies_by(examples, fn {_t, label} -> label end)
Logger.info("Using #{length(examples)} examples  #{inspect(dist)}")

{train, test} = Dataset.split(examples, 0.8, seed: seed)
Logger.info("Split: #{length(train)} train / #{length(test)} test")

Logger.info("Training (alpha=#{alpha}, balance=#{balance})...")
{us, model} = :timer.tc(fn -> ToxicClassifier.NaiveBayes.train(train, alpha: alpha, balance: balance) end)
Logger.info("Trained in #{Float.round(us / 1_000, 1)} ms  (vocab: #{model.vocab_size} features)")

classify_fn = fn m, text -> ToxicClassifier.NaiveBayes.classify(m, text, threshold: threshold) end

Logger.info("Evaluating on #{length(test)} docs (parallel across #{System.schedulers_online()} cores)...")
{ev_us, metrics} = :timer.tc(fn -> Evaluation.evaluate(model, test, classify_fn, :toxic) end)
Logger.info("Evaluated in #{Float.round(ev_us / 1_000, 1)} ms")

Evaluation.report(metrics)

IO.puts("── Sample predictions (threshold #{threshold}) ─────")

demo = [
  "you are an absolute idiot and everyone hates you",
  "thanks so much, this really helped me out",
  "shut up you brainless troll nobody asked",
  "looking forward to the meetup next week",
  "what a stupid worthless piece of garbage",
  "could you please share the documentation link"
]

for text <- demo do
  proba = ToxicClassifier.NaiveBayes.predict_proba(model, text)
  label = ToxicClassifier.NaiveBayes.classify(model, text, threshold: threshold)
  tox = Map.get(proba, :toxic, 0.0)
  tag = if label == :toxic, do: "TOXIC", else: "clean"
  IO.puts("  [#{String.pad_trailing(tag, 5)} #{:erlang.float_to_binary(tox * 100, decimals: 1)}%]  #{text}")
end

Logger.flush()
