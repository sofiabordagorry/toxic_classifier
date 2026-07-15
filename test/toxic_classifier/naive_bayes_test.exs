defmodule ToxicClassifier.NaiveBayesTest do
  use ExUnit.Case, async: true

  alias ToxicClassifier.NaiveBayes

  @examples [
    {"you are an idiot and everyone hates you", :toxic},
    {"shut up you stupid moron", :toxic},
    {"what a worthless piece of garbage", :toxic},
    {"thanks so much for the help", :clean},
    {"have a lovely day everyone", :clean},
    {"could you share the documentation", :clean}
  ]

  test "classifies obvious toxic and clean text" do
    model = NaiveBayes.train(@examples)

    assert NaiveBayes.predict(model, "what an idiot you are") == :toxic
    assert NaiveBayes.predict(model, "thanks for the lovely help") == :clean
  end

  test "predict_proba returns a normalised distribution" do
    model = NaiveBayes.train(@examples)
    proba = NaiveBayes.predict_proba(model, "you stupid moron")

    assert_in_delta proba.toxic + proba.clean, 1.0, 1.0e-9
    assert proba.toxic > proba.clean
  end

  test "threshold makes classification stricter" do
    model = NaiveBayes.train(@examples)

    # A very high threshold should be harder to trip as toxic.
    assert NaiveBayes.classify(model, "idiot", threshold: 0.5) == :toxic
    assert NaiveBayes.classify(model, "have a nice documentation day", threshold: 0.9) == :clean
  end

  test "min_count pruning shrinks the vocabulary" do
    full = NaiveBayes.train(@examples, min_count: 1)
    pruned = NaiveBayes.train(@examples, min_count: 2)

    assert pruned.vocab_size < full.vocab_size
  end
end
