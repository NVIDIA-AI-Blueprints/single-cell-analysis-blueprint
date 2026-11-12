import json
import sys
from pathlib import Path

MARKERS = ("do_shutdown(True)", "IPython.Application.instance()")

def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: ci_strip_kernel_shutdown.py <input.ipynb> <output.ipynb>",
            file=sys.stderr,
        )
        return 2
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    nb = json.loads(inp.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    kept = []
    removed = 0
    for c in cells:
        if c.get("cell_type") != "code":
            kept.append(c)
            continue
        src = "".join(c.get("source", []))
        if all(m in src for m in MARKERS):
            removed += 1
            continue
        kept.append(c)
    nb["cells"] = kept
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False)
        f.write("\n")
    print(f"ci_strip_kernel_shutdown: removed {removed} shutdown cell(s) -> {outp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
