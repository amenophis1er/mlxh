#!/bin/bash
# Install mlxh into $MLXH_HOME (default ~/.mlxh) and a launcher into $MLXH_BIN
# (default ~/.local/bin). Requires uv (https://docs.astral.sh/uv/).
set -euo pipefail

PREFIX="${MLXH_HOME:-$HOME/.mlxh}"
BIN="${MLXH_BIN:-$HOME/.local/bin}"
SRC="$(cd "$(dirname "$0")" && pwd)"

command -v uv >/dev/null || { echo "error: uv is required (brew install uv)"; exit 1; }

echo "Installing mlxh into $PREFIX"
mkdir -p "$PREFIX" "$BIN"
rm -rf "$PREFIX/app"
cp -R "$SRC/app" "$PREFIX/app"

if [ ! -x "$PREFIX/venv/bin/python" ]; then
  uv venv --python 3.11 "$PREFIX/venv"
fi
uv pip install -p "$PREFIX/venv/bin/python" -r "$PREFIX/app/requirements.txt"

cat > "$BIN/mlxh" <<EOF
#!/bin/bash
export MLXH_HOME="$PREFIX"
export MLXH_LAUNCHER="$BIN/mlxh"
exec "$PREFIX/venv/bin/python" "$PREFIX/app/mlxh.py" "\$@"
EOF
chmod +x "$BIN/mlxh"

echo
echo "Installed. Launcher: $BIN/mlxh"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "note: $BIN is not on your PATH — add it, or call the launcher by full path" ;;
esac
echo "Try:  mlxh pull mlx-community/Qwen2.5-0.5B-Instruct-4bit && mlxh chat Qwen2.5-0.5B-Instruct-4bit"
