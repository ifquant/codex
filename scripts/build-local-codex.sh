#!/usr/bin/env bash
# Build the custom macOS CLI package, reusing the native dev-small cache.
set -euo pipefail

case "${1:-}" in
  --help|-h)
    echo "Usage: $0 [--clean-cache]"
    echo "Build target/local-package; optionally remove compilation caches after success."
    echo "Run without concurrent Cargo jobs. Existing top-level executables are kept."
    exit 0 ;;
  ""|--clean-cache) ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Too many arguments" >&2; exit 2; }
[[ "$(uname -s)" == Darwin ]] || { echo "This local build script supports macOS." >&2; exit 2; }

repo=$(cd "$(dirname "$0")/.." && pwd)
target="$repo/codex-rs/target"
output="$target/local-package"
backup="$target/.local-package-previous"
[[ ! -e "$backup" ]] || { echo "Recover the previous package at $backup first." >&2; exit 1; }
mkdir -p "$target"
staging=$(mktemp -d "$target/.local-package.XXXXXX")
trap 'rm -rf "$staging"' EXIT

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export V8_FROM_SOURCE="${V8_FROM_SOURCE:-1}"
export GN_ARGS="${GN_ARGS:-is_debug=false}"
if [[ -z "${LIBCLANG_PATH:-}" ]]; then
  export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
fi

cd "$repo/codex-rs"
cargo build --target-dir "$target" --profile dev-small \
  -p codex-cli -p codex-code-mode-host
cd "$repo"
just assemble-codex-package \
  --entrypoint-bin "$target/dev-small/codex" \
  --code-mode-host-bin "$target/dev-small/codex-code-mode-host" \
  --package-dir "$staging"
"$staging/bin/codex" --version

if [[ -e "$output" ]]; then mv "$output" "$backup"; fi
if ! mv "$staging" "$output"; then
  if [[ -e "$backup" ]]; then mv "$backup" "$output"; fi
  exit 1
fi
rm -rf "$backup"

if [[ "${1:-}" == --clean-cache ]]; then
  # Keep top-level binaries that running CLI/helper processes may still need.
  rm -rf "$target/dev-small/deps" "$target/dev-small/build" \
    "$target/dev-small/.fingerprint" "$target/dev-small/incremental"
fi
printf '\nPackage ready: %s/bin/codex\nStart daemon: %s/bin/codex app-server daemon start\n' "$output" "$output"
