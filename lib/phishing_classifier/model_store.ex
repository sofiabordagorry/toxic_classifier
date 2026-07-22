defmodule PhishingClassifier.ModelStore do
  @moduledoc """
  Holds the currently trained models in memory, globally, so they survive page
  reloads and are shared across everyone using the app until someone retrains.

  On boot it rehydrates from the on-disk cache (see
  `PhishingClassifier.ModelBundle`), so a trained model also survives a server
  restart — no need to re-upload the dataset every time.
  """

  use Agent

  alias PhishingClassifier.ModelBundle

  def start_link(_opts) do
    Agent.start_link(&load_persisted/0, name: __MODULE__)
  end

  @doc "Returns the stored models map, or `nil` if nothing has been trained yet."
  def get, do: Agent.get(__MODULE__, & &1)

  def put(models), do: Agent.update(__MODULE__, fn _ -> models end)

  def clear do
    ModelBundle.delete()
    Agent.update(__MODULE__, fn _ -> nil end)
  end

  # Rebuild the UI model map from the persisted bundle, or start empty.
  defp load_persisted do
    case ModelBundle.load() do
      nil -> nil
      bundle -> ModelBundle.to_ui(bundle)
    end
  end
end
