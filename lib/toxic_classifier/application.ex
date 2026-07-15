defmodule ToxicClassifier.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Phoenix.PubSub, name: ToxicClassifier.PubSub},
      ToxicClassifier.ModelStore,
      ToxicClassifierWeb.Endpoint
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: ToxicClassifier.Supervisor)
  end

  @impl true
  def config_change(changed, _new, removed) do
    ToxicClassifierWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
