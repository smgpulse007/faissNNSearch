"""Render actual demo/test command output. Optional dependency: Pillow."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]


def font(size):
    for name in ("DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def main():
    frames, durations, commands = [], [], []
    for args in (["demo.py"], ["-m", "unittest", "discover", "-s", "tests"]):
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", timeout=60)
        if result.returncode:
            raise RuntimeError("Recording command failed; no success animation written")
        command = "python " + " ".join(args)
        output = result.stdout + result.stderr
        commands.append({"command": command, "exit_code": result.returncode, "output": output})
        pages = []
        for block in output.strip().split("\n\n"):
            lines = [part for line in block.splitlines() for part in (textwrap.wrap(line, 88) or [""])]
            pages.extend(lines[start:start + 12] for start in range(0, len(lines), 12))
        # Each page contains only output actually returned by the subprocess.
        for page in pages:
            for count in range(1, len(page) + 1):
                frame = Image.new("RGB", (1040, 500), "#0c1220")
                draw = ImageDraw.Draw(frame)
                draw.rounded_rectangle((18, 18, 1022, 482), radius=16, fill="#111d30", outline="#2b3f59", width=2)
                draw.text((42, 40), "FAISS / exact vector search", font=font(22), fill="#edf5ff")
                draw.text((42, 88), "$ " + command, font=font(17), fill="#7dd3fc")
                for row, line in enumerate(page[:count]):
                    draw.text((42, 130 + row * 24), line, font=font(17), fill="#e1e9f5")
                draw.text((42, 451), "Actual command output  |  Synthetic vectors  |  CPU", font=font(14), fill="#91a4c0")
                frames.append(frame)
                durations.append(3000 if count == len(page) else 140)
    target = ROOT / "docs"
    target.mkdir(exist_ok=True)
    frames[0].save(target / "search-demo.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    sources = ["vector_search.py", "search.py", "demo.py", "tests/test_search.py",
               "examples/vectors.json", "examples/queries.json", "requirements.txt", "scripts/record_demo.py"]
    metadata = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                "kind": "animation_rendered_from_actual_command_output", "pillow_version": pillow_version,
                "text_hash_normalization": "UTF-8 text with CRLF normalized to LF",
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest() for name in sources},
                "commands": commands}
    (target / "demo-recording.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Recorded actual demo and tests in docs/search-demo.gif")


if __name__ == "__main__":
    main()
