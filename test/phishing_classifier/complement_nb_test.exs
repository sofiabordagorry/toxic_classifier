defmodule PhishingClassifier.ComplementNBTest do
  use ExUnit.Case, async: true

  alias PhishingClassifier.{NaiveBayes, ComplementNB}

  @examples [
    {"verify your account now or it will be permanently suspended", :phish},
    {"urgent: confirm your password immediately, nobody else can", :phish},
    {"you have won a prize, click here to claim your reward now", :phish},
    {"update your billing details to avoid service interruption", :phish},
    {"thanks so much for the help today", :safe},
    {"have a lovely day everyone", :safe},
    {"could you share the documentation link", :safe},
    {"the weather is nice for a walk", :safe}
  ]

  test "classifies phishing and safe text via the complement" do
    cnb = @examples |> NaiveBayes.train(min_count: 1) |> ComplementNB.from_model()

    assert ComplementNB.classify(cnb, "verify your password now") == :phish
    assert ComplementNB.classify(cnb, "have a lovely documentation day") == :safe
  end

  test "predict_proba is normalised and discriminative without L1 flattening" do
    cnb = @examples |> NaiveBayes.train(min_count: 1) |> ComplementNB.from_model()

    phish = ComplementNB.predict_proba(cnb, "confirm your suspended account password")
    safe = ComplementNB.predict_proba(cnb, "thanks for the lovely help")

    assert_in_delta phish.phish + phish.safe, 1.0, 1.0e-9
    # Probabilities should be confidently separated, not clustered near 0.5.
    assert phish.phish > 0.7
    assert safe.phish < 0.3
  end
end
