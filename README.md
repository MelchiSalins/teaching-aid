# teaching-aid

Manim animations for teaching ML / systems concepts (matmul, LLM dataflow, …).

## Quickstart (macOS)

Prereqs: [Homebrew](https://brew.sh) and [uv](https://github.com/astral-sh/uv).

```bash
make install    # installs brew deps, creates .venv via uv, installs LaTeX packages
make smoke      # end-to-end verification (renders a LaTeX scene)
```

Run `make` (or `make help`) to see every available target.

## Common commands

```bash
make render-sample   # render manim_scripts/sample_scene.py::HelloManim
make lint            # ruff check
make format          # ruff format
make doctor          # print versions of the whole toolchain
make clean           # remove media/ and caches
```

Or render any scene directly:

```bash
uv run manim -pql llm_dataflow.py LLMDataFlow     # fast preview
uv run manim -pqh llm_dataflow.py LLMDataFlow     # 1080p60 teaching quality
```

## VS Code

Open the folder and accept the recommended extensions. The workspace is
pre-configured to use the project-local `./.venv` interpreter — no manual
selection required.

## Docs

- [`_docs/manim_setup.md`](_docs/manim_setup.md) — setup deep dive & troubleshooting
- [`_docs/llm_dataflow_animation.md`](_docs/llm_dataflow_animation.md)
- [`_docs/matmul_animation.md`](_docs/matmul_animation.md)
