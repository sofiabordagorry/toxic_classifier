defmodule ToxicClassifier.Bert do
  @moduledoc """
  Wraps pretrained transformer classifiers (via Bumblebee) as optional models to
  compare against. Several can be registered; each loads in the background at
  startup, so the app stays usable while they warm up (or if they never load,
  e.g. on a small host). Set `ENABLE_BERT=false` to skip them entirely.
  """

  use GenServer
  require Logger

  @topic "models"

  @registry [
    %{
      id: :general,
      name: "toxic-bert",
      repo: {:hf, "unitary/toxic-bert"},
      tokenizer: {:hf, "bert-base-uncased"},
      scores_function: :sigmoid,
      toxic_labels: ["toxic"],
      desc: "General toxicity, trained on Wikipedia comments (Jigsaw).",
      meta: "BERT · English · ~440 MB"
    },
    %{
      id: :gaming,
      name: "gaming BERT",
      repo: {:hf, "yehort/distilbert-gaming-chat-toxicity-en"},
      tokenizer: {:hf, "distilbert-base-uncased"},
      scores_function: :softmax,
      toxic_labels: ["LABEL_1", "toxic", "1"],
      desc: "Fine-tuned on gaming chat — knows slang like “gg ez”, “noob”, “uninstall”.",
      meta: "DistilBERT · English · ~256 MB"
    },
    %{
      id: :multilingual,
      name: "multilingual",
      repo: {:hf, "unitary/multilingual-toxic-xlm-roberta"},
      tokenizer: {:hf, "FacebookAI/xlm-roberta-base"},
      scores_function: :sigmoid,
      toxic_labels: ["toxic"],
      desc: "toxic-bert's multilingual sibling (XLM-RoBERTa) — works in Spanish and many other languages.",
      meta: "XLM-RoBERTa · multilingual · ~1.1 GB"
    }
  ]

  def start_link(_opts), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @doc "List of %{id, name, desc, meta} for the UI selector."
  def options, do: Enum.map(@registry, &Map.take(&1, [:id, :name, :desc, :meta]))

  def default_id, do: :general

  def name(id), do: config(id).name

  @doc "Status of a model: :loading | :ready | :failed | :disabled."
  def status(id), do: GenServer.call(__MODULE__, {:status, id})

  @doc "P(toxic) in 0..1 for the given model, or nil if it isn't ready."
  def toxic_score(id, text), do: GenServer.call(__MODULE__, {:score, id, text}, 30_000)

  @impl true
  def init(:ok) do
    if enabled?() do
      parent = self()
      Enum.each(@registry, fn m -> Task.start(fn -> send(parent, {:loaded, m.id, load_serving(m)}) end) end)
      {:ok, %{servings: %{}, statuses: Map.new(@registry, &{&1.id, :loading})}}
    else
      {:ok, %{servings: %{}, statuses: Map.new(@registry, &{&1.id, :disabled})}}
    end
  end

  @impl true
  def handle_info({:loaded, id, {:ok, serving}}, state) do
    Logger.info("bert #{id} ready")
    Phoenix.PubSub.broadcast(ToxicClassifier.PubSub, @topic, {:bert_ready, id})
    {:noreply, %{state | servings: Map.put(state.servings, id, serving), statuses: Map.put(state.statuses, id, :ready)}}
  end

  def handle_info({:loaded, id, {:error, reason}}, state) do
    Logger.error("bert #{id} failed to load: #{inspect(reason)}")
    {:noreply, %{state | statuses: Map.put(state.statuses, id, :failed)}}
  end

  @impl true
  def handle_call({:status, id}, _from, state), do: {:reply, Map.get(state.statuses, id, :disabled), state}

  def handle_call({:score, id, text}, _from, state) do
    case Map.get(state.servings, id) do
      nil ->
        {:reply, nil, state}

      serving ->
        toxic_labels = config(id).toxic_labels
        %{predictions: preds} = Nx.Serving.run(serving, text)
        toxic = Enum.find(preds, &(&1.label in toxic_labels))
        {:reply, toxic && toxic.score, state}
    end
  end

  defp config(id), do: Enum.find(@registry, &(&1.id == id))

  defp load_serving(m) do
    with {:ok, model} <- Bumblebee.load_model(m.repo),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(m.tokenizer) do
      serving =
        Bumblebee.Text.text_classification(model, tokenizer,
          scores_function: m.scores_function,
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
