defmodule ToxicClassifier.Bert do
  @moduledoc """
  Wraps the pretrained `unitary/toxic-bert` transformer (via Bumblebee) as an
  optional fourth classifier.
  """

  use GenServer
  require Logger

  @repo {:hf, "unitary/toxic-bert"}
  @topic "models"

  def start_link(_opts), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @doc "One of :loading | :ready | :failed | :disabled."
  def status, do: GenServer.call(__MODULE__, :status)

  @doc "P(toxic) in 0..1, or nil if the model isn't ready."
  def toxic_score(text), do: GenServer.call(__MODULE__, {:score, text}, 30_000)

  @impl true
  def init(:ok) do
    if enabled?() do
      parent = self()
      Task.start(fn -> send(parent, {:load_result, load_serving()}) end)
      {:ok, %{serving: nil, status: :loading}}
    else
      {:ok, %{serving: nil, status: :disabled}}
    end
  end

  @impl true
  def handle_info({:load_result, {:ok, serving}}, state) do
    Logger.info("toxic-bert ready")
    Phoenix.PubSub.broadcast(ToxicClassifier.PubSub, @topic, :bert_ready)
    {:noreply, %{state | serving: serving, status: :ready}}
  end

  def handle_info({:load_result, {:error, reason}}, state) do
    Logger.error("toxic-bert failed to load: #{inspect(reason)}")
    {:noreply, %{state | status: :failed}}
  end

  @impl true
  def handle_call(:status, _from, state), do: {:reply, state.status, state}

  def handle_call({:score, _text}, _from, %{serving: nil} = state), do: {:reply, nil, state}

  def handle_call({:score, text}, _from, %{serving: serving} = state) do
    %{predictions: preds} = Nx.Serving.run(serving, text)
    toxic = Enum.find(preds, &(&1.label == "toxic"))
    {:reply, toxic && toxic.score, state}
  end

  defp load_serving do
    with {:ok, model} <- Bumblebee.load_model(@repo),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer({:hf, "bert-base-uncased"}) do
      serving =
        Bumblebee.Text.text_classification(model, tokenizer,
          scores_function: :sigmoid,
          compile: [batch_size: 1, sequence_length: 64],
          defn_options: [compiler: EXLA]
        )

      {:ok, serving}
    end
  rescue
    e -> {:error, e}
  end

  defp enabled?, do: System.get_env("ENABLE_BERT", "true") != "false"
end
