# Run Notes (Local)

This file captures the exact local setup so startup is fast next time.

## One-command start/stop

From repo root:

```bash
./_scripts/dev_up.sh
```

Stop services:

```bash
./_scripts/dev_down.sh
```

## What `dev_up.sh` does

1. Reads environment variables from `../config.txt` (PowerShell format: `$env:KEY="VALUE"`).
2. Exports runtime vars for backend + frontend.
3. Starts backend on `127.0.0.1:8080`.
4. Starts frontend on `127.0.0.1:3000` with `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8080`.
5. Writes logs and pid files under `.run/`.

## Important paths

- Backend log: `.run/backend.log`
- Frontend log: `.run/frontend.log`
- Backend pid : `.run/backend.pid`
- Frontend pid: `.run/frontend.pid`

## Prereqs that are now installed on this machine

- Python 3.11 (Homebrew)
- `libpq` (Homebrew, needed by `psycopg2` build/runtime)
- Poetry (user install under `~/Library/Python/3.9/bin/poetry`)
- Frontend dependencies in `frontend/node_modules`

## Known caveat (Weaviate)

Current `WEAVIATE_URL` in `../config.txt` returns `404` for `v1/meta`, so startup previously failed.
A safe fallback is in place in backend code to keep the app bootable when Weaviate is unavailable.
When a valid Weaviate endpoint is provided, retrieval will work normally.

## Quick health checks

```bash
curl -sS http://127.0.0.1:8080/docs >/dev/null && echo backend-ok
curl -sS http://127.0.0.1:3000 >/dev/null && echo frontend-ok
```
