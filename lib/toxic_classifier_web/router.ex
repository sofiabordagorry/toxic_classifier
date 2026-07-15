defmodule ToxicClassifierWeb.Router do
  use ToxicClassifierWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {ToxicClassifierWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  scope "/", ToxicClassifierWeb do
    pipe_through :browser

    live "/", ClassifierLive
  end
end
