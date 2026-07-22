defmodule PhishingClassifier.MixProject do
  use Mix.Project

  def project do
    [
      app: :phishing_classifier,
      version: "0.1.0",
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      mod: {PhishingClassifier.Application, []},
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nimble_csv, "~> 1.2"},
      {:phoenix, "~> 1.7"},
      {:phoenix_live_view, "~> 1.0"},
      {:phoenix_html, "~> 4.1"},
      {:bandit, "~> 1.5"},
      {:jason, "~> 1.4"},
      {:bumblebee, "~> 0.6"},
      {:exla, "~> 0.9"}
    ]
  end
end
