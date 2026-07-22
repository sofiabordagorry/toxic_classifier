defmodule PhishingClassifier.NaiveBayesTest do
  use ExUnit.Case, async: true

  alias PhishingClassifier.NaiveBayes

  @examples [
    {"verify your account now or it will be suspended", :phish},
    {"urgent: confirm your password immediately to avoid closure", :phish},
    {"you have won a prize, click here to claim your reward", :phish},
    {"thanks so much for the help", :safe},
    {"have a lovely day everyone", :safe},
    {"could you share the documentation", :safe}
  ]

  test "classifies obvious phishing and safe text" do
    model = NaiveBayes.train(@examples)

    assert NaiveBayes.predict(model, "confirm your password to verify your account") == :phish
    assert NaiveBayes.predict(model, "thanks for the lovely help") == :safe
  end

  test "predict_proba returns a normalised distribution" do
    model = NaiveBayes.train(@examples)
    proba = NaiveBayes.predict_proba(model, "verify your password now")

    assert_in_delta proba.phish + proba.safe, 1.0, 1.0e-9
    assert proba.phish > proba.safe
  end

  test "threshold makes classification stricter" do
    model = NaiveBayes.train(@examples)

    # A very high threshold should be harder to trip as phishing.
    assert NaiveBayes.classify(model, "verify your account", threshold: 0.5) == :phish
    assert NaiveBayes.classify(model, "have a nice documentation day", threshold: 0.9) == :safe
  end

  test "min_count pruning shrinks the vocabulary" do
    full = NaiveBayes.train(@examples, min_count: 1)
    pruned = NaiveBayes.train(@examples, min_count: 2)

    assert pruned.vocab_size < full.vocab_size
  end
end
