defmodule PhishingClassifier.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Phoenix.PubSub, name: PhishingClassifier.PubSub},
      PhishingClassifier.ModelStore,
      PhishingClassifier.Bert,
      PhishingClassifierWeb.Endpoint
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: PhishingClassifier.Supervisor)
  end

  @impl true
  def config_change(changed, _new, removed) do
    PhishingClassifierWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
