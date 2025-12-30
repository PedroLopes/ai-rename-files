#!/usr/bin/env bash
set -e

TOOL_NAME="ai-rename-files"
TOOL_FILE="ai-rename-files.py"
INSTALL_DIR="."
#INSTALL_DIR="$HOME/.local/$TOOL_NAME"
BIN_DIR="$HOME/.local/bin"

echo "Installing $TOOL_NAME..."

# Ensure directories exist
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Create virtual environment
python3 -m venv "$INSTALL_DIR/venv"

# Activate and install deps
source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Copy script
cp "$TOOL_FILE" "$INSTALL_DIR/$TOOL_NAME"

# Create launcher
cat > "$BIN_DIR/$TOOL_NAME" <<EOF
#!/usr/bin/env bash
source "$INSTALL_DIR/venv/bin/activate"
exec python "$INSTALL_DIR/$TOOL_NAME" "\$@"
EOF

chmod +x "$BIN_DIR/$TOOL_NAME"

read -p "Add ~/.local/bin to your PATH? [y/N] " answer
[[ "$answer" != "y" ]] && exit 0

SHELL_NAME=$(basename "$SHELL")

case "$SHELL_NAME" in
  bash)
    PROFILE="$HOME/.bashrc"
    ;;
  zsh)
    PROFILE="$HOME/.zshrc"
    ;;
  *)
    echo "Unknown shell. Please add ~/.local/bin to PATH manually."
    exit 0
    ;;
esac

LINE='export PATH="$HOME/.local/bin:$PATH"'

grep -qxF "$LINE" "$PROFILE" || echo "$LINE" >> "$PROFILE"

echo "Added to $PROFILE. Restart your shell."

echo "Installed successfully!"
