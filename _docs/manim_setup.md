# Manim Local Development Setup (macOS)

This doc is the deep-dive companion to the top-level [`Makefile`](../Makefile)
and [`README.md`](../README.md). If you just want to get going, run:

```bash
make install
make smoke
```

Everything below explains *what that does* and how to debug it when it
doesn't.

---

## 1. The toolchain

| Layer                 | Tool            | Why                                                       |
| --------------------- | --------------- | --------------------------------------------------------- |
| Python version mgmt   | **uv**          | Reads `.python-version` (3.11) and provisions `.venv/`    |
| Python deps           | **uv + `uv.lock`** | Reproducible install of `manim==0.20.1` and dev tools  |
| 2-D graphics          | `cairo` (brew)  | Underlies `pycairo`                                       |
| Text layout           | `pango` (brew)  | Underlies `manimpango`                                    |
| Video encoding        | `ffmpeg` (brew) | Manim shells out for MP4/GIF output                       |
| Build glue            | `pkgconf` (brew)| Lets native Python wheels find cairo/pango at build time  |
| LaTeX rendering       | **BasicTeX** (cask) | `MathTex`, `Matrix`, any math — shells out to `latex`  |
| Editor                | VS Code         | Pre-configured in `.vscode/` to use `./.venv`             |
| Lint / format         | `ruff`          | Installed as a uv dev dependency                          |

> **Why `uv` and not conda?** `uv` is ~10× faster, resolves deterministically
> from `uv.lock`, and doesn't need a base install of Miniconda. The old
> conda-based workflow has been fully retired.

---

## 2. Prerequisites

Install these **once** on the machine; everything else is driven by `make`.

```bash
# Homebrew (if you don't already have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sanity check: `make check-tools` should print ✓ for `brew`, `uv`, `make`.

---

## 3. One-shot install

```bash
make install
```

Under the hood this runs, in order:

1. **`make check-tools`** — verifies `brew`, `uv`, `make` exist.
2. **`make brew-deps`** — `brew install ffmpeg cairo pango pkgconf` and
   `brew install --cask basictex`. Idempotent (skips already-installed
   casks).
3. **`make venv`** — `uv sync`, which:
   - Reads `.python-version` (3.11) and downloads/uses that interpreter.
   - Creates `./.venv/`.
   - Installs exactly what's in `uv.lock`.
   - Builds `pycairo` **from source** (forced via
     `tool.uv.no-binary-package = ["pycairo"]` in `pyproject.toml`) so it
     links against the Homebrew arm64 cairo on Apple Silicon.
4. **`make tex-deps`** — `sudo tlmgr update --self && sudo tlmgr install …`
   with the package set Manim needs (`standalone`, `preview`,
   `doublestroke`, `physics`, `dvisvgm`, `xcolor`, …). Prompts for your
   macOS password.

After it finishes, open a new terminal (or `eval "$(/usr/libexec/path_helper)"`)
so `latex` is on `PATH`, then:

```bash
make smoke   # renders latex_smoke.py::LatexSmoke
```

A green ✓ means the full toolchain — uv, manim, cairo, pango, ffmpeg,
LaTeX — is healthy.

---

## 4. Why pycairo is built from source

The prebuilt `pycairo` wheels on PyPI for some releases are x86_64-only. On
Apple Silicon you'd see:

```
ImportError: … (mach-o file, but is an incompatible architecture
(have 'x86_64', need 'arm64' …))
```

We side-step this by telling `uv` never to use the wheel:

```toml
# pyproject.toml
[tool.uv]
no-binary-package = ["pycairo"]
```

Combined with `brew install cairo pkgconf`, the source build produces a
native arm64 binary.

---

## 5. VS Code integration

`.vscode/settings.json` pins the interpreter to the workspace-local venv:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true
}
```

`.vscode/extensions.json` recommends:

- **Python** (`ms-python.python`)
- **Pylance** (`ms-python.vscode-pylance`)
- **Ruff** (`charliermarsh.ruff`)

When you open the folder, VS Code offers to install them. If the
interpreter isn't auto-detected: `Cmd+Shift+P` → *Python: Select
Interpreter* → pick the one under `./.venv/`.

---

## 6. Rendering & CLI cheat-sheet

All commands can be prefixed with `uv run ` (which uses `./.venv` automatically)
or run after `source .venv/bin/activate`.

| Command                                                  | What it does                     |
| -------------------------------------------------------- | -------------------------------- |
| `uv run manim -pql file.py SceneName`                    | Preview, low (480p15) quality    |
| `uv run manim -pqh file.py SceneName`                    | Preview, high (1080p60) quality  |
| `uv run manim -pqk file.py SceneName`                    | Preview, 4k (2160p60) quality    |
| `uv run manim -s file.py SceneName`                      | Save final frame as PNG          |
| `uv run manim --format=gif -ql file.py SceneName`        | Render to animated GIF           |
| `uv run manim cfg show`                                  | Show resolved configuration      |
| `make render-sample`                                     | Shortcut for `HelloManim` scene  |
| `make smoke`                                             | Shortcut for `LatexSmoke` scene  |

Rendered artifacts land under `./media/` (gitignored). `make clean` wipes it.

---

## 7. Updating dependencies

```bash
# Bump a single package (respects pyproject constraints):
uv lock --upgrade-package manim

# Upgrade everything in the resolver:
uv lock --upgrade

# Apply the new lockfile to .venv:
uv sync
```

Commit both `pyproject.toml` and `uv.lock` together.

---

## 8. Troubleshooting

| Symptom                                                    | Fix                                                                                 |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `ImportError … 'x86_64', need 'arm64'` on pycairo          | `uv sync --reinstall-package pycairo` (we force source builds — brew cairo must be installed first) |
| `FileNotFoundError: 'latex'` after install                 | `eval "$(/usr/libexec/path_helper)"` or open a new terminal                         |
| `! LaTeX Error: File 'standalone.cls' not found.`          | `make tex-deps` (re-runs the `sudo tlmgr install …` list)                           |
| `pdfTeX error: pdflatex (file cmr10): can't find format…`  | `sudo fmtutil-sys --all`                                                            |
| `dvisvgm: error: cannot run program`                       | `make tex-deps` — `dvisvgm` is in the list                                          |
| TeX repo appears stale                                     | `sudo tlmgr option repository https://mirror.aarnet.edu.au/pub/CTAN/systems/texlive/tlnet` |
| `uv: command not found`                                    | `curl -LsSf https://astral.sh/uv/install.sh \| sh` then open a new shell            |
| Everything seems broken                                    | `make doctor` — prints versions of every layer in the stack                         |

If in doubt, nuke and rebuild:

```bash
make distclean      # removes media/ and .venv
make install
make smoke
```
