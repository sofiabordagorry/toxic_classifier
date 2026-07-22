defmodule PhishingClassifier.Bert do
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
      id: :phishing,
      name: "phishing-BERT",
      repo: {:hf, "ealvaradob/bert-finetuned-phishing"},
      tokenizer: {:hf, "bert-large-uncased"},
      scores_function: :softmax,
      positive_labels: ["phishing"],
      desc: "bert-large fine-tuned to detect phishing across emails, SMS and URLs.",
      meta: "BERT-large · 336M params · English · ~1.3 GB"
    }
  ]

  def start_link(_opts), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @doc "List of %{id, name, desc, meta} for the UI selector."
  def options, do: Enum.map(@registry, &Map.take(&1, [:id, :name, :desc, :meta]))

  def default_id, do: :phishing

  def name(id), do: config(id).name

  @doc "Status of a model: :loading | :ready | :failed | :disabled."
  def status(id), do: GenServer.call(__MODULE__, {:status, id})

  @doc "P(phishing) in 0..1 for the given model, or nil if it isn't ready."
  def phish_score(id, text), do: GenServer.call(__MODULE__, {:score, id, text}, 30_000)

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
    Phoenix.PubSub.broadcast(PhishingClassifier.PubSub, @topic, {:bert_ready, id})
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
        positive_labels = config(id).positive_labels
        %{predictions: preds} = Nx.Serving.run(serving, text)
        positive = Enum.find(preds, &(&1.label in positive_labels))
        {:reply, positive && positive.score, state}
    end
  end

  defp config(id), do: Enum.find(@registry, &(&1.id == id))

  defp load_serving(m) do
    with {:ok, model} <- Bumblebee.load_model(m.repo),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(m.tokenizer) do
      serving =
        Bumblebee.Text.text_classification(model, tokenizer,
          scores_function: m.scores_function,
          # Bucketed lengths: short messages use the 64 bucket (fast), long
          # emails use up to 512 (BERT's max) so the payload isn't truncated.
          compile: [batch_size: 1, sequence_length: [64, 256, 512]],
          defn_options: [compiler: EXLA]
        )

      {:ok, serving}
    end
  rescue
    e -> {:error, e}
  end

  defp enabled?, do: System.get_env("ENABLE_BERT", "true") != "false"
end
