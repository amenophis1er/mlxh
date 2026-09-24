#!/bin/bash
# Install mlxh into $MLXH_HOME (default ~/.mlxh) and a launcher into $MLXH_BIN
# (default ~/.local/bin). Requires uv (https://docs.astral.sh/uv/).
#
# Works two ways:
#   from a checkout:  ./install.sh
#   without a clone:  curl -fsSL https://raw.githubusercontent.com/amenophis1er/mlxh/main/install.sh | bash
set -euo pipefail

REPO="amenophis1er/mlxh"
PREFIX="${MLXH_HOME:-$HOME/.mlxh}"
BIN="${MLXH_BIN:-$HOME/.local/bin}"
SRC="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"

command -v uv >/dev/null || { echo "error: uv is required (brew install uv)"; exit 1; }

if [ -z "$SRC" ] || [ ! -f "$SRC/pyproject.toml" ]; then
  # Not running from a checkout (e.g. curl | bash): fetch the repo tarball.
  command -v curl >/dev/null || { echo "error: curl is required"; exit 1; }
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  AUTH=()
  [ -n "${GITHUB_TOKEN:-}" ] && AUTH=(-H "Authorization: token $GITHUB_TOKEN")
  echo "Fetching $REPO ..."
  curl -fsSL "${AUTH[@]}" "https://api.github.com/repos/$REPO/tarball/main" \
    | tar -xz -C "$TMP"
  SRC="$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [ -f "$SRC/pyproject.toml" ] || { echo "error: download did not contain pyproject.toml"; exit 1; }
fi

echo "Installing mlxh into $PREFIX"
mkdir -p "$PREFIX" "$BIN"
rm -rf "$PREFIX/app"  # layout from pre-package versions

if [ ! -x "$PREFIX/venv/bin/python" ]; then
  uv venv --python 3.11 "$PREFIX/venv"
fi
uv pip install -p "$PREFIX/venv/bin/python" --reinstall-package mlxh "$SRC"

# An existing optional image runtime carries its own mlxh code. Synchronize
# after every upgrade; startup checks refuse stale code if this repair fails.
if [ -e "$PREFIX/images/current" ]; then
  MLXH_HOME="$PREFIX" "$PREFIX/venv/bin/mlxh" images install || \
    echo "Image runtime needs repair: mlxh images install"
fi

cat > "$BIN/mlxh" <<EOF
#!/bin/bash
export MLXH_HOME="$PREFIX"
export MLXH_LAUNCHER="$BIN/mlxh"
exec "$PREFIX/venv/bin/mlxh" "\$@"
EOF
chmod +x "$BIN/mlxh"

echo
echo "Installed. Launcher: $BIN/mlxh"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "note: $BIN is not on your PATH — add it, or call the launcher by full path" ;;
esac
echo "Try:  mlxh pull mlx-community/Qwen2.5-0.5B-Instruct-4bit && mlxh chat Qwen2.5-0.5B-Instruct-4bit"
