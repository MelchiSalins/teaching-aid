# LLM Data-Flow Animation

A Manim teaching animation that walks an IT-professional audience through
exactly what happens inside a decoder-only LLM when it sees a word — all the
way to populating the **KV cache**.

Scene file: [`manim_scripts/llm_dataflow.py`](../manim_scripts/llm_dataflow.py)
Scene class: `LLMDataFlow`

## What it covers (in order)

1. **Tokenize** — sentence `"The cat sat down"` → integer IDs
2. **Embedding** — token ID indexes a row of the learned embedding matrix
3. **+ Positional** — element-wise add of a position vector so the model
   knows word order (`x_i = e_i + p_i`)
4. **Q / K / V** — each `x_i` is projected through three weight matrices
   `W_Q, W_K, W_V` to produce `q_i, k_i, v_i`
5. **Attention** — `softmax( Q · Kᵀ / √d ) · V`, shown as an iteration over
   queries `q_1, q_2, q_3` (causal mask visible: each token can only see
   itself and earlier tokens)
6. **KV Cache** — 3 autoregressive generation steps. Each new token only
   computes its own `q, k, v`; old `k, v` are reused from the cache.
   Finishes with the O(n²) → O(n) punchline.

## LaTeX is used for math

Formulas (`Attention = softmax(QKᵀ/√d) · V`, subscripts like `q_i`,
`e_cat`, big-O notation, etc.) and all vectors/matrices are rendered with
`MathTex` / `Matrix`, which requires a working LaTeX install. BasicTeX is
enough — see **§7** of [`manim_setup.md`](manim_setup.md) for exact
install commands. The `latex_smoke.py` scene is the quickest way to
confirm your LaTeX toolchain is wired up before running the full teaching
animation.


## Rendering

Activate the env first:

```bash
conda activate manim-env
```

Then from the repo root:

| Purpose                         | Command                                                           |
| ------------------------------- | ----------------------------------------------------------------- |
| Fast preview (~30 s, 480p15)    | `manim -pql manim_scripts/llm_dataflow.py LLMDataFlow`            |
| Teaching-grade (~3-5 min, 1080p60) | `manim -pqh manim_scripts/llm_dataflow.py LLMDataFlow`         |
| 4K master                       | `manim -pqk manim_scripts/llm_dataflow.py LLMDataFlow`            |
| Animated GIF (shorter/lower-q)  | `manim --format=gif -ql manim_scripts/llm_dataflow.py LLMDataFlow`|

Output lands at:

```
media/videos/llm_dataflow/<resolution>/LLMDataFlow.mp4
```

A successful low-quality render (480p15) produced a ~3.5 MB file with 134
animations and no errors — that's the smoke test.

## Visual vocabulary

To keep the story readable across 8 sub-scenes, colours are used consistently:

| Colour   | Meaning                           |
| -------- | --------------------------------- |
| BLUE     | Query (`q_i`) / positional vector |
| GREEN    | Key (`k_i`)                       |
| ORANGE   | Value (`v_i`)                     |
| YELLOW   | Currently-active / highlighted    |
| GREY     | Cached / reused / masked out      |

## Tweaking

Common customisations (all near the top of the file):

- **Sentence**: change `SENTENCE` and `TOKEN_IDS` (length is derived).
- **Vector width**: `D_MODEL` (kept at 4 for readability; don't push past ~6).
- **Which attention iterations to show**: search for `for q_idx in [1, 2, 3]`
  inside `_attention()` and adjust the list.
- **Number of generation steps** for the KV-cache demo: edit `step_tokens`
  / `step_indices` in `_kv_cache()`.

## Intended audience & teaching notes

- Built for **IT professionals** who already know what a matrix is but have
  never cracked open a transformer.
- Each sub-scene has a caption at the bottom so the video can be paused and
  read as static frames during a talk.
- The roadmap pill-bar at the top lights up the current step in yellow and
  greys out future steps — viewers always know "where we are in the
  pipeline".
- The final O(n²) → O(n) punchline is the motivating *why* for anyone who
  has ever wondered why LLM inference is so much faster than training.
