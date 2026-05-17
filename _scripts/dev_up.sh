#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/../config.txt}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/bin"

POETRY_BIN="${POETRY_BIN:-}"
if [[ -z "${POETRY_BIN}" ]]; then
  if command -v poetry >/dev/null 2>&1; then
    POETRY_BIN="$(command -v poetry)"
  elif [[ -x "${HOME}/Library/Python/3.9/bin/poetry" ]]; then
    POETRY_BIN="${HOME}/Library/Python/3.9/bin/poetry"
  else
    echo "poetry not found. Set POETRY_BIN or add poetry to PATH." >&2
    exit 1
  fi
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -x "${ROOT_DIR}/_scripts/dev_down.sh" ]]; then
  echo "missing _scripts/dev_down.sh" >&2
  exit 1
fi

load_env_from_config() {
  local file="$1"
  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*\$env:([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*\"([^\"]*)\" ]]; then
      export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
    fi
  done < "$file"
}

# Next.js may shell out to yarn when yarn.lock exists. Provide a tiny local shim.
cat > "${RUN_DIR}/bin/yarn" <<'YARN'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" || "${1:-}" == "-v" ]]; then
  echo "1.22.19"
  exit 0
fi
if [[ "${1:-}" == "install" ]]; then
  shift
  exec npm install "$@"
fi
exec npm "$@"
YARN
chmod +x "${RUN_DIR}/bin/yarn"

"${ROOT_DIR}/_scripts/dev_down.sh" >/dev/null 2>&1 || true

load_env_from_config "${CONFIG_FILE}"

export PATH="${RUN_DIR}/bin:/opt/homebrew/opt/libpq/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8080"

(
  cd "${ROOT_DIR}"
  nohup "${POETRY_BIN}" run uvicorn backend.main:app --host 127.0.0.1 --port 8080 >"${RUN_DIR}/backend.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/backend.pid"
)

(
  cd "${ROOT_DIR}/frontend"
  nohup npm run dev -- --hostname 127.0.0.1 --port 3000 >"${RUN_DIR}/frontend.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/frontend.pid"
)

wait_for_http() {
  local url="$1"
  local name="$2"
  for _ in {1..45}; do
    if curl -sS "$url" >/dev/null 2>&1; then
      echo "${name} is up: ${url}"
      return 0
    fi
    sleep 1
  done
  echo "${name} did not become ready: ${url}" >&2
  return 1
}

wait_for_http "http://127.0.0.1:8080/docs" "backend"
wait_for_http "http://127.0.0.1:3000" "frontend"

echo ""
echo "Started successfully."
echo "Frontend: http://127.0.0.1:3000"
echo "Backend : http://127.0.0.1:8080/docs"
echo "Logs    : ${RUN_DIR}/frontend.log and ${RUN_DIR}/backend.log"
echo "Stop    : ${ROOT_DIR}/_scripts/dev_down.sh"
