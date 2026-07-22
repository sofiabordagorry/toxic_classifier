import Config

config :logger, :default_handler, config: [type: :standard_error]

config :logger, :default_formatter, format: "$time [$level] $message\n"

config :phishing_classifier, PhishingClassifierWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  http: [ip: {127, 0, 0, 1}, port: 4000],
  render_errors: [formats: [html: PhishingClassifierWeb.ErrorHTML], layout: false],
  pubsub_server: PhishingClassifier.PubSub,
  live_view: [signing_salt: "toxSALT01"],
  secret_key_base: "kZq3xY8pR2mVnB6wLtF4jH9sC1dG7aE0uI5oP3rT8yW2qX6zN4bM1vK9cJ7hD0lS"

config :phoenix, :json_library, Jason

config :nx, default_backend: EXLA.Backend
