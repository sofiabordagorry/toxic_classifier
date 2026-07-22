defmodule PhishingClassifierWeb.Layouts do
  use PhishingClassifierWeb, :html

  def root(assigns) do
    ~H"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="csrf-token" content={Phoenix.Controller.get_csrf_token()} />
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <title>Phishing Detector</title>
        <style>
          :root {
            --accent: #7c2130;
            --accent-hover: #641622;
            --accent-soft: #f7ecee;
            --ink: #1c1917;
            --muted: #78716c;
            --line: #e7e5e4;
            color-scheme: light;
          }
          * { box-sizing: border-box; }
          body {
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
            margin: 0; padding: 4rem 1.25rem;
            background: #faf9f8; color: var(--ink);
            display: flex; justify-content: center;
            -webkit-font-smoothing: antialiased;
          }
          .wrap { width: 100%; max-width: 480px; }
          .header { text-align: center; margin-bottom: 1.75rem; }
          h1 { font-size: 1.45rem; font-weight: 700; margin: 0; letter-spacing: -.02em; }
          .sub { color: var(--muted); margin: .5rem auto 0; font-size: .92rem; max-width: 34ch; line-height: 1.5; }
          .card {
            background: #fff; border: 1px solid var(--line); border-radius: 14px;
            padding: 1.4rem; margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
          }
          .about-lead { margin: 0 0 1.1rem; color: #57534e; font-size: .92rem; line-height: 1.55; }
          .models { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: .95rem; }
          .models li { display: flex; flex-direction: column; gap: .15rem; padding-left: .85rem; border-left: 2px solid var(--accent); }
          .m-name { font-weight: 650; font-size: .9rem; color: var(--ink); }
          .m-desc { font-size: .85rem; color: var(--muted); line-height: 1.5; }
          .m-desc a { color: var(--accent); font-weight: 600; text-decoration: none; white-space: nowrap; }
          .m-desc a:hover { text-decoration: underline; }
          label { display: block; font-size: .82rem; font-weight: 600; color: #57534e; margin-bottom: .35rem; }
          .upload-note { margin: 0 0 .9rem; font-size: .82rem; color: var(--muted); }
          .upload-note code { background: #f5f5f4; padding: .1rem .35rem; border-radius: 5px; font-size: .78rem; color: #44403c; }
          .drop {
            border: 1.5px dashed var(--line); border-radius: 11px; padding: 1rem;
            margin-bottom: 1rem; background: #faf9f8; transition: border-color .15s;
          }
          .drop:hover { border-color: #d6b3ba; }
          input[type=file] { width: 100%; font-size: .88rem; color: #57534e; }
          input[type=file]::file-selector-button {
            background: var(--accent-soft); color: var(--accent); border: 0; border-radius: 8px;
            padding: .5rem .9rem; margin-right: .8rem; font-weight: 600; cursor: pointer;
            font-size: .85rem;
          }
          input[type=file]::file-selector-button:hover { filter: brightness(.97); }
          button {
            width: 100%;
            background: var(--accent); color: #fff; border: 0; border-radius: 10px;
            padding: .75rem 1.2rem; font-size: .95rem; font-weight: 600; cursor: pointer;
            transition: background .15s;
          }
          button:hover { background: var(--accent-hover); }
          button:disabled { opacity: .55; cursor: default; }
          textarea {
            width: 100%; background: #fff; color: var(--ink);
            border: 1px solid #d6d3d1; border-radius: 10px; padding: .8rem;
            font-size: 1rem; font-family: inherit; resize: vertical; line-height: 1.5;
            transition: border-color .15s, box-shadow .15s;
          }
          textarea:focus {
            outline: none; border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(124,33,48,.12);
          }
          .muted { color: var(--muted); font-size: .9rem; }
          .summary {
            color: #57534e; font-size: .86rem; margin: 0 0 1.2rem;
            padding: .55rem .8rem; background: #f5f5f4; border: 1px solid var(--line);
            border-radius: 9px;
          }
          .err { color: var(--accent); background: var(--accent-soft); border: 1px solid #ecd5d9; padding: .8rem 1rem; border-radius: 10px; font-size: .9rem; }
          .meters { display: flex; flex-direction: column; gap: 1.05rem; margin-top: 1.3rem; }
          .meter-head {
            display: flex; justify-content: space-between; align-items: center;
            font-size: .9rem; margin-bottom: .4rem;
          }
          .meter-head .name { font-weight: 600; color: #292524; }
          .meter-head .right { display: flex; align-items: center; gap: .55rem; }
          .meter-head .val { font-variant-numeric: tabular-nums; font-weight: 700; min-width: 42px; text-align: right; }
          .tag { font-size: .68rem; font-weight: 700; letter-spacing: .03em; text-transform: uppercase; padding: .12rem .45rem; border-radius: 999px; }
          .tag.phishing { background: var(--accent-soft); color: var(--accent); }
          .tag.safe { background: #eef4ee; color: #3f7d54; }
          .bar { height: 9px; background: #eeecea; border-radius: 999px; overflow: hidden; }
          .fill { height: 100%; border-radius: 999px; background: var(--accent); transition: width .14s ease; }
          .hint { color: #a8a29e; font-size: .82rem; margin-top: 1.2rem; text-align: center; }
          .bert-picker { margin-top: 1.3rem; }
          .picker-label { display: block; font-size: .8rem; font-weight: 600; color: #57534e; margin-bottom: .5rem; }
          .segmented { display: inline-flex; gap: .45rem; flex-wrap: wrap; }
          button.seg {
            width: auto; background: #fff; color: #57534e; border: 1px solid var(--line);
            border-radius: 9px; padding: .5rem .9rem; font-size: .85rem; font-weight: 600;
            position: relative; cursor: pointer;
          }
          button.seg:hover { background: #faf9f8; }
          button.seg.on { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
          .seg .tip {
            display: none; position: absolute; z-index: 10; left: 0; top: calc(100% + .5rem);
            width: 240px; background: #1c1917; color: #fafafa; border-radius: 10px;
            padding: .7rem .8rem; font-weight: 400; text-align: left; line-height: 1.45;
            box-shadow: 0 8px 24px rgba(0,0,0,.2);
          }
          .seg:hover .tip { display: block; }
          .tip b { display: block; font-size: .85rem; margin-bottom: .15rem; }
          .tip-meta { display: block; font-size: .72rem; color: #a8a29e; margin-bottom: .4rem; }
          .tip-desc { display: block; font-size: .8rem; color: #e7e5e4; }
        </style>
      </head>
      <body>
        {@inner_content}
        <script src="/assets/phoenix.js"></script>
        <script src="/assets/phoenix_live_view.js"></script>
        <script>
          let csrfToken = document.querySelector("meta[name='csrf-token']").getAttribute("content")
          let liveSocket = new LiveView.LiveSocket("/live", Phoenix.Socket, {params: {_csrf_token: csrfToken}})
          liveSocket.connect()
        </script>
      </body>
    </html>
    """
  end
end
