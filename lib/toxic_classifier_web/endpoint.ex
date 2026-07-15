defmodule ToxicClassifierWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :toxic_classifier

  @session_options [
    store: :cookie,
    key: "_toxic_classifier_key",
    signing_salt: "toxSESS01",
    same_site: "Lax"
  ]

  socket "/live", Phoenix.LiveView.Socket,
    websocket: [connect_info: [session: @session_options]]

  plug Plug.Static,
    at: "/",
    from: :toxic_classifier,
    gzip: false,
    only: ToxicClassifierWeb.static_paths()

  plug Plug.RequestId
  plug Plug.Telemetry, event_prefix: [:phoenix, :endpoint]

  plug Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Phoenix.json_library()

  plug Plug.MethodOverride
  plug Plug.Head
  plug Plug.Session, @session_options
  plug ToxicClassifierWeb.Router
end
