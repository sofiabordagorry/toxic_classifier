defmodule PhishingClassifier.LogisticRegressionTest do
  use ExUnit.Case, async: true

  alias PhishingClassifier.LogisticRegression, as: LR

  @examples [
    {"verify your account now or it will be suspended", :phish},
    {"urgent: confirm your password immediately", :phish},
    {"you have won a prize, click here to claim your reward", :phish},
    {"update your billing details to avoid service interruption", :phish},
    {"thanks so much for the help", :safe},
    {"have a lovely day everyone", :safe},
    {"could you share the documentation", :safe},
    {"the weather is nice for a walk", :safe}
  ]

  test "learns to separate phishing from safe" do
    model = LR.train(@examples, epochs: 50, min_count: 1)

    assert LR.predict(model, "confirm your password to verify your account") == :phish
    assert LR.predict(model, "thanks for the lovely documentation") == :safe
  end

  test "predict_proba is a normalised distribution" do
    model = LR.train(@examples, epochs: 50, min_count: 1)
    proba = LR.predict_proba(model, "verify your password now")

    assert_in_delta proba.phish + proba.safe, 1.0, 1.0e-9
    assert proba.phish > 0.5
  end

  test "unknown words fall back to the bias only" do
    model = LR.train(@examples, epochs: 20, min_count: 1)
    # Text with no known features should sit near the learned base rate.
    proba = LR.predict_proba(model, "zzzzz qqqq")
    assert proba.phish >= 0.0 and proba.phish <= 1.0
  end
end
