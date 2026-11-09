# Avatar Studio — the merged base-video + avatar page

## What it is

`web/avatar.html` used to be "Avatar Creator" (upload → genavatar) and
`web/enhance.html` was "Base video studio" (matte, lighting, GFPGAN, 9:16
canvas, background swap). They are now ONE page at `/avatar.html`; `enhance.html`
is a redirect stub so old links keep working.

The merge is intentionally NOT a new wizard widget: there is one form whose
meaning is decided by the "Enhance before creating" switch (default on) plus
whether an avatar ID is filled in.

| Switch | Avatar ID | Result |
|--------|-----------|--------|
| on  | set   | `POST /api/enhance/task` + `avatar_model`/`avatar_id`/`avatar_params` → enhance job, chained genavatar on completion |
| on  | empty | enhance-only job (the old base-video studio, same job list/detail) |
| off | set   | `POST /api/avatar/task`, straight genavatar (the old creator) |
| off | empty | blocked client-side with a message |

## Backend pieces

- `server/avatar_routes.py`
  - `AVATAR_GENERATORS` — the registry: model id → label + parameter specs
    (`name/label/type/default/min/max/options/hint`). Serve it via
    `GET /api/avatar/models`; the page renders the model select AND the
    "Advanced" inputs from it. **Never hardcode model options in HTML** — add a
    backend here and the UI offers it.
  - `build_task_params()` validates/coerces (int/bool/select/text, `pads` must
    be four ints) and raises a message the route returns verbatim.
  - `enqueue_avatar_task()` is the single creation path, used by
    `/api/avatar/task`, the enhance chain, and `/api/enhance/task/{id}/avatar`.
- `server/enhance_routes.py`
  - `EnhanceJob.params["avatar"]` holds the chain spec;
    `EnhanceJobManager._chain_avatar()` enqueues genavatar when the enhance job
    completes and stores `avatar_task_id` / `avatar_error` on the job, so the
    link survives a page reload. Chain failures never fail the enhance job.
  - `_parse_avatar_spec()` accepts either nested `avatar` JSON or the flat
    multipart fields (`avatar_model`, `avatar_id`, `avatar_params`).
- `server/task_manager.py` — avatar tasks expose `source_job` so the UI can show
  "from #<enhance id>".

## Frontend notes

- One merged task list (in-memory dicts from `/api/enhance/tasks` and
  `/api/avatar/tasks`) with All / Enhance / Avatars filters; selecting a row
  shows the matching detail card (video preview, swap, create-from-result for
  enhance jobs; progress + kv for avatar tasks).
- Guard detail input fields behind a "selection changed" key — a 1.5 s poll must
  not stomp the user's typing.
- Avatar IDs are free text with a `<datalist>` of `GET /api/avatars` plus a
  confirm on overwrite.

## Verification recipe (no model weights needed)

Server: `./restart.sh --skip-model --listenport 8099` (background it — a
foreground restart gets killed by the caller's timeout and takes the server
down). Only ONE instance may listen on the port: several instances restart each
other and silently wipe the in-memory task lists.

- `tests/test_avatar_registry.py` — registry/model-param/spec validation.
- UI harness (jsdom, in-repo page + live server) — model select from registry,
  params re-render on model change, merged list + filters, detail panes,
  datalist, validation. Pattern: load `web/avatar.html` with
  `runScripts:"dangerously"`, replace `window.fetch` to proxy to the server,
  replace `window.FormData` with Node's (jsdom's is not accepted by fetch).
- E2E: drive the page's own submit handler for both flows and assert the
  server-side chain (`avatar_task_id` on the job, `source_job` on the task).
