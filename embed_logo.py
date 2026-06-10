"""One-shot: bake nav logo into all HTML pages as a data URI."""
import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
logo = HERE / "assets" / "nauticai-logo-nav.png"
src = "data:image/png;base64," + base64.b64encode(logo.read_bytes()).decode()

for f in HERE.glob("*.html"):
    text = f.read_text(encoding="utf-8")
    new = re.sub(r'src="/assets/nauticai-logo[^"]*"', f'src="{src}"', text)
    new = re.sub(r'src="data:image/png;base64,[^"]+"', f'src="{src}"', new)
    if new != text:
        f.write_text(new, encoding="utf-8")
        print("updated", f.name)

print("logo embedded (%d chars)" % len(src))
