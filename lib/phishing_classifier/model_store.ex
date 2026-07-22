defmodule PhishingClassifier.ModelStore do
  @moduledoc """
  Holds the currently trained models in memory, globally, so they survive page
  reloads and are shared across everyone using the app until someone retrains.
  """

  use Agent

  def start_link(_opts) do
    Agent.start_link(fn -> nil end, name: __MODULE__)
  end

  @doc "Returns the stored models map, or `nil` if nothing has been trained yet."
  def get, do: Agent.get(__MODULE__, & &1)

  def put(models), do: Agent.update(__MODULE__, fn _ -> models end)

  def clear, do: Agent.update(__MODULE__, fn _ -> nil end)
end
