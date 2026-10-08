#!/usr/bin/env bash
# Build docs/CardioVision-Technical-Documentation.pdf from technical-documentation.md.
# Needs pandoc and Google Chrome / Chromium. Regenerate figures first with docs/make_figures.py.
set -euo pipefail
cd "$(dirname "$0")"
pandoc technical-documentation.md --standalone --self-contained --css doc-style.css \
  --metadata pagetitle="CardioVision technical documentation" -o technical-documentation.html
CHROME=$(command -v google-chrome || command -v chromium || command -v chromium-browser)
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --user-data-dir="$(mktemp -d)" \
  --print-to-pdf=CardioVision-Technical-Documentation.pdf "file://$PWD/technical-documentation.html" 2>/dev/null
rm -f technical-documentation.html
n=$(pdfinfo CardioVision-Technical-Documentation.pdf | awk '/^Pages:/ {print $2}')
echo "CardioVision-Technical-Documentation.pdf: $n pages$([ "$n" -le 6 ] || echo '  (OVER the 6-page limit)')"
