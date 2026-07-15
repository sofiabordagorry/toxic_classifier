defmodule ToxicClassifier.ComplementNBTest do
  use ExUnit.Case, async: true

  alias ToxicClassifier.{NaiveBayes, ComplementNB}

  @examples [
    {"you are an idiot and everyone hates you", :toxic},
    {"shut up you stupid moron nobody asked", :toxic},
    {"what a worthless piece of garbage you are", :toxic},
    {"go away you pathetic disgusting loser", :toxic},
    {"thanks so much for the help today", :clean},
    {"have a lovely day everyone", :clean},
    {"could you share the documentation link", :clean},
    {"the weather is nice for a walk", :clean}
  ]

  test "classifies toxic and clean text via the complement" do
    cnb = @examples |> NaiveBayes.train(min_count: 1) |> ComplementNB.from_model()

    assert ComplementNB.classify(cnb, "you stupid idiot") == :toxic
    assert ComplementNB.classify(cnb, "have a lovely documentation day") == :clean
  end

  test "predict_proba is normalised and discriminative without L1 flattening" do
    cnb = @examples |> NaiveBayes.train(min_count: 1) |> ComplementNB.from_model()

    toxic = ComplementNB.predict_proba(cnb, "worthless garbage loser")
    clean = ComplementNB.predict_proba(cnb, "thanks for the lovely help")

    assert_in_delta toxic.toxic + toxic.clean, 1.0, 1.0e-9
    # Probabilities should be confidently separated, not clustered near 0.5.
    assert toxic.toxic > 0.7
    assert clean.toxic < 0.3
  end
end
