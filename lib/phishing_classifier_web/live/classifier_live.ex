defmodule PhishingClassifierWeb.ClassifierLive do
  use PhishingClassifierWeb, :live_view

  alias PhishingClassifier.{Dataset, ModelStore, NaiveBayes, ComplementNB, LogisticRegression}

  @topic "models"

  @models_info [
    {"Naive Bayes",
     "Counts how often each word shows up in phishing vs. safe emails — the classic text-classification baseline.",
     nil},
    {"Complement NB",
     "A Naive Bayes variant that estimates each class from its complement, coping better with imbalanced data.",
     nil},
    {"Logistic Regression",
     "Learns a weight per word with gradient descent instead of assuming words are independent — usually the most accurate.",
     nil},
    {"Pretrained transformer",
     "A 336M-parameter BERT fine-tuned for phishing. Understands context, but is thousands of times heavier than the models above. Runs in Elixir via",
     {"Bumblebee ↗", "https://hexdocs.pm/bumblebee"}}
  ]

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket), do: Phoenix.PubSub.subscribe(PhishingClassifier.PubSub, @topic)

    {:ok,
     socket
     |> assign(
       models: ModelStore.get(),
       training?: false,
       train_error: nil,
       text: "",
       results: [],
       bert_id: PhishingClassifier.Bert.default_id(),
       bert_options: PhishingClassifier.Bert.options(),
       bert_status: PhishingClassifier.Bert.status(PhishingClassifier.Bert.default_id())
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
    results = score(socket.assigns.models, text, socket.assigns.bert_id)
    {:noreply, assign(socket, text: text, results: results)}
  end

  @impl true
  def handle_event("select_bert", %{"bert" => id}, socket) do
    bert_id = String.to_existing_atom(id)
    results = score(socket.assigns.models, socket.assigns.text, bert_id)

    {:noreply,
     assign(socket,
       bert_id: bert_id,
       bert_status: PhishingClassifier.Bert.status(bert_id),
       results: results
     )}
  end

  @impl true
  def handle_async(:training, {:ok, models}, socket) do
    ModelStore.put(models)

    Phoenix.PubSub.broadcast_from(
      PhishingClassifier.PubSub,
      self(),
      @topic,
      {:models_updated, models}
    )

    {:noreply,
     assign(socket,
       models: models,
       training?: false,
       results: score(models, socket.assigns.text, socket.assigns.bert_id)
     )}
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
    results = score(models, socket.assigns.text, socket.assigns.bert_id)
    {:noreply, assign(socket, models: models, results: results)}
  end

  @impl true
  def handle_info({:bert_ready, _id}, socket) do
    results = score(socket.assigns.models, socket.assigns.text, socket.assigns.bert_id)

    {:noreply,
     assign(socket,
       bert_status: PhishingClassifier.Bert.status(socket.assigns.bert_id),
       results: results
     )}
  end

  @impl true
  def render(assigns) do
    assigns = assign(assigns, :models_info, @models_info)

    ~H"""
    <div class="wrap">
      <div class="header">
        <h1>Phishing Email Detector</h1>
      </div>

      <div class="card about">
        <p class="about-lead">
          Upload a labelled CSV and three classic classifiers train on it, then compare them
          against a pretrained transformer. Paste any message and each one scores how likely it's phishing.
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
          <label>Paste a message</label>
          <textarea name="text" rows="3" phx-debounce="120"
                    placeholder="e.g. Your account has been suspended. Verify your identity now..."><%= @text %></textarea>
        </form>

        <div class="bert-picker">
          <span class="picker-label">Transformer to compare</span>
          <div class="segmented">
            <button :for={opt <- @bert_options} type="button"
                    class={["seg", opt.id == @bert_id && "on"]}
                    phx-click="select_bert" phx-value-bert={opt.id}>
              {opt.name}
              <span class="tip">
                <b>{opt.name}</b>
                <span class="tip-meta">{opt.meta}</span>
                <span class="tip-desc">{opt.desc}</span>
              </span>
            </button>
          </div>
        </div>

        <div class="meters">
          <div :for={{name, p} <- @results} class="meter">
            <div class="meter-head">
              <span class="name">{name}</span>
              <span class="right">
                <span class={"tag #{if p >= 0.5, do: "phishing", else: "safe"}"}>
                  {if p >= 0.5, do: "phishing", else: "safe"}
                </span>
                <span class="val">{pct(p)}</span>
              </span>
            </div>
            <div class="bar"><div class="fill" style={"width:#{p * 100}%"}></div></div>
          </div>
        </div>

        <p :if={@results == []} class="hint">Start typing above to see each model's phishing score.</p>
        <p :if={@bert_status == :loading} class="hint">{PhishingClassifier.Bert.name(@bert_id)} is warming up…</p>
        <p :if={@bert_status == :failed} class="hint">{PhishingClassifier.Bert.name(@bert_id)} couldn't load on this host.</p>
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
    dest = Path.join(System.tmp_dir!(), "phishing_upload_#{System.unique_integer([:positive])}.csv")
    File.cp!(path, dest)
    {:ok, dest}
  end

  defp train_models(path) do
    examples = Dataset.load(path)

    nb = NaiveBayes.train(examples, balance_ratio: 1.0, min_count: 2)
    cnb = ComplementNB.from_model(nb)

    lr =
      LogisticRegression.train(examples,
        balance_ratio: 1.0,
        min_count: 2,
        epochs: 8,
        positive: :phish,
        negative: :safe
      )

    classifiers = [
      {"Naive Bayes", &Map.get(NaiveBayes.predict_proba(nb, &1), :phish, 0.0)},
      {"Complement NB", &Map.get(ComplementNB.predict_proba(cnb, &1), :phish, 0.0)},
      {"Logistic Regression", &Map.get(LogisticRegression.predict_proba(lr, &1), :phish, 0.0)}
    ]

    dist = Enum.frequencies_by(examples, fn {_text, label} -> label end)

    summary =
      "Trained on #{length(examples)} examples · " <>
        "#{Map.get(dist, :phish, 0)} phishing / #{Map.get(dist, :safe, 0)} safe"

    %{classifiers: classifiers, summary: summary}
  end

  defp score(_models, "", _bert_id), do: []
  defp score(nil, _text, _bert_id), do: []

  defp score(%{classifiers: classifiers}, text, bert_id) do
    Enum.map(classifiers, fn {name, proba_fn} -> {name, proba_fn.(text)} end) ++
      bert_result(text, bert_id)
  end

  defp bert_result(text, bert_id) do
    if PhishingClassifier.Bert.status(bert_id) == :ready do
      case PhishingClassifier.Bert.phish_score(bert_id, text) do
        nil -> []
        score -> [{PhishingClassifier.Bert.name(bert_id), score}]
      end
    else
      []
    end
  end

  defp pct(p), do: "#{round(p * 100)}%"
end
