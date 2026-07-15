defmodule ToxicClassifierWeb.ClassifierLive do
  use ToxicClassifierWeb, :live_view

  alias ToxicClassifier.{Dataset, ModelStore, NaiveBayes, ComplementNB, LogisticRegression}

  @topic "models"

  @models_info [
    {"Naive Bayes",
     "Counts how often each word shows up in toxic vs. clean comments — the classic text-classification baseline.",
     nil},
    {"Complement NB",
     "A Naive Bayes variant that estimates each class from its complement, coping better with imbalanced data.",
     nil},
    {"Logistic Regression",
     "Learns a weight per word with gradient descent instead of assuming words are independent — usually the most accurate.",
     nil},
    {"toxic-bert",
     "A pretrained transformer that understands context — it catches cases the simple models miss, at the cost of being much heavier. Runs in Elixir via",
     {"Bumblebee ↗", "https://hexdocs.pm/bumblebee"}}
  ]

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket), do: Phoenix.PubSub.subscribe(ToxicClassifier.PubSub, @topic)

    {:ok,
     socket
     |> assign(
       models: ModelStore.get(),
       training?: false,
       train_error: nil,
       text: "",
       results: [],
       bert_status: ToxicClassifier.Bert.status()
     )
     |> allow_upload(:dataset, accept: ~w(.csv), max_entries: 1, max_file_size: 200_000_000)}
  end

  @impl true
  def handle_event("validate", _params, socket), do: {:noreply, socket}

  @impl true
  def handle_event("train", _params, socket) do
    case consume_uploaded_entries(socket, :dataset, &copy_upload/2) do
      [path] ->
        {:noreply,
         socket
         |> assign(training?: true, train_error: nil)
         |> start_async(:training, fn -> train_models(path) end)}

      [] ->
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("classify", %{"text" => text}, socket) do
    {:noreply, assign(socket, text: text, results: score(socket.assigns.models, text))}
  end

  @impl true
  def handle_async(:training, {:ok, models}, socket) do
    ModelStore.put(models)

    Phoenix.PubSub.broadcast_from(
      ToxicClassifier.PubSub,
      self(),
      @topic,
      {:models_updated, models}
    )

    {:noreply,
     assign(socket, models: models, training?: false, results: score(models, socket.assigns.text))}
  end

  @impl true
  def handle_async(:training, {:exit, _reason}, socket) do
    {:noreply,
     assign(socket,
       training?: false,
       train_error: "Training failed — check the CSV has a text and a label column."
     )}
  end

  @impl true
  def handle_info({:models_updated, models}, socket) do
    {:noreply, assign(socket, models: models, results: score(models, socket.assigns.text))}
  end

  @impl true
  def handle_info(:bert_ready, socket) do
    {:noreply,
     assign(socket,
       bert_status: :ready,
       results: score(socket.assigns.models, socket.assigns.text)
     )}
  end

  @impl true
  def render(assigns) do
    assigns = assign(assigns, :models_info, @models_info)

    ~H"""
    <div class="wrap">
      <div class="header">
        <h1>Toxic Comment Classifier</h1>
      </div>

      <div class="card about">
        <p class="about-lead">
          Upload a labelled CSV and three classic classifiers train on it, then compare them
          against a pretrained transformer. Type any phrase and each one scores how toxic it looks.
        </p>
        <ul class="models">
          <li :for={{name, desc, link} <- @models_info}>
            <span class="m-name">{name}</span>
            <span class="m-desc">
              {desc}<a :if={link} href={elem(link, 1)} target="_blank" rel="noopener">{" "}{elem(link, 0)}</a>
            </span>
          </li>
        </ul>
      </div>

      <div :if={@training?} class="card muted">Training on your dataset…</div>
      <div :if={@train_error} class="card err">{@train_error}</div>

      <div :if={@models && !@training?} class="card">
        <p class="summary">{@models.summary}</p>

        <form phx-change="classify">
          <label>Type a phrase</label>
          <textarea name="text" rows="3" phx-debounce="120"
                    placeholder="e.g. you are so..."><%= @text %></textarea>
        </form>

        <div class="meters">
          <div :for={{name, p} <- @results} class="meter">
            <div class="meter-head">
              <span class="name">{name}</span>
              <span class="right">
                <span class={"tag #{if p >= 0.5, do: "toxic", else: "clean"}"}>
                  {if p >= 0.5, do: "toxic", else: "clean"}
                </span>
                <span class="val">{pct(p)}</span>
              </span>
            </div>
            <div class="bar"><div class="fill" style={"width:#{p * 100}%"}></div></div>
          </div>
        </div>

        <p :if={@results == []} class="hint">Start typing above to see each model's toxicity score.</p>
        <p :if={@bert_status == :loading} class="hint">toxic-bert (transformer) is warming up…</p>
      </div>

      <form class="card" phx-change="validate" phx-submit="train">
        <label>{if @models, do: "Retrain on a new dataset", else: "Train the models"}</label>
        <p class="upload-note">A labelled CSV — <code>text,label</code> or the Jigsaw format.</p>
        <div class="drop">
          <.live_file_input upload={@uploads.dataset} />
        </div>
        <button type="submit" disabled={@training?}>
          {if @models, do: "Retrain", else: "Train"}
        </button>
      </form>
    </div>
    """
  end

  defp copy_upload(%{path: path}, _entry) do
    dest = Path.join(System.tmp_dir!(), "toxic_upload_#{System.unique_integer([:positive])}.csv")
    File.cp!(path, dest)
    {:ok, dest}
  end

  defp train_models(path) do
    examples = Dataset.load(path)

    nb = NaiveBayes.train(examples, balance_ratio: 1.0, min_count: 2)
    cnb = ComplementNB.from_model(nb)
    lr = LogisticRegression.train(examples, balance_ratio: 1.0, min_count: 2, epochs: 8)

    classifiers = [
      {"Naive Bayes", &Map.get(NaiveBayes.predict_proba(nb, &1), :toxic, 0.0)},
      {"Complement NB", &Map.get(ComplementNB.predict_proba(cnb, &1), :toxic, 0.0)},
      {"Logistic Regression", &Map.get(LogisticRegression.predict_proba(lr, &1), :toxic, 0.0)}
    ]

    dist = Enum.frequencies_by(examples, fn {_text, label} -> label end)

    summary =
      "Trained on #{length(examples)} examples · " <>
        "#{Map.get(dist, :toxic, 0)} toxic / #{Map.get(dist, :clean, 0)} clean"

    %{classifiers: classifiers, summary: summary}
  end

  defp score(_models, ""), do: []
  defp score(nil, _text), do: []

  defp score(%{classifiers: classifiers}, text) do
    Enum.map(classifiers, fn {name, proba_fn} -> {name, proba_fn.(text)} end) ++ bert_result(text)
  end

  defp bert_result(text) do
    if ToxicClassifier.Bert.status() == :ready do
      case ToxicClassifier.Bert.toxic_score(text) do
        nil -> []
        score -> [{"toxic-bert", score}]
      end
    else
      []
    end
  end

  defp pct(p), do: "#{round(p * 100)}%"
end
