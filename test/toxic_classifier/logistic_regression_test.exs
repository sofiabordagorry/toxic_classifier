defmodule ToxicClassifier.LogisticRegressionTest do
  use ExUnit.Case, async: true

  alias ToxicClassifier.LogisticRegression, as: LR

  @examples [
    {"you are an idiot and everyone hates you", :toxic},
    {"shut up you stupid moron", :toxic},
    {"what a worthless piece of garbage", :toxic},
    {"go away you pathetic loser", :toxic},
    {"thanks so much for the help", :clean},
    {"have a lovely day everyone", :clean},
    {"could you share the documentation", :clean},
    {"the weather is nice for a walk", :clean}
  ]

  test "learns to separate toxic from clean" do
    model = LR.train(@examples, epochs: 50, min_count: 1)

    assert LR.predict(model, "what an idiot you are") == :toxic
    assert LR.predict(model, "thanks for the lovely documentation") == :clean
  end

  test "predict_proba is a normalised distribution" do
    model = LR.train(@examples, epochs: 50, min_count: 1)
    proba = LR.predict_proba(model, "you stupid moron")

    assert_in_delta proba.toxic + proba.clean, 1.0, 1.0e-9
    assert proba.toxic > 0.5
  end

  test "unknown words fall back to the bias only" do
    model = LR.train(@examples, epochs: 20, min_count: 1)
    # Text with no known features should sit near the learned base rate.
    proba = LR.predict_proba(model, "zzzzz qqqq")
    assert proba.toxic >= 0.0 and proba.toxic <= 1.0
  end
end
