# Matrix Multiplication: GPU vs. Systolic Array

A Manim teaching animation that walks the viewer through the **same 3×3 matrix
multiply** on three different pieces of hardware:

1. A human with a pencil
2. A CPU (one scalar ALU)
3. An NVIDIA-style GPU warp (SIMT, one thread per output cell)
4. A systolic array (AWS Trainium NeuronCore style, output-stationary)

The point is not to teach matmul — it's to show *why* specialised AI
accelerators exist, using numbers the viewer has just computed in their head.

Scene file: [`manim_scripts/matmul_gpu_vs_systolic.py`](../manim_scripts/matmul_gpu_vs_systolic.py)
Scene class: `MatmulGPUvsSystolic`

## The worked example

```
A = [1 2 3]    B = [1 0 2]        C = A·B = [ 7  5 13]
    [4 5 6]        [0 1 1]                  [16 11 31]
    [7 8 9]        [2 1 3]                  [25 17 49]
```

27 multiply-adds (MACs) in total — small enough for anyone to verify on
paper. The same workload is traced through every architecture so the
viewer can do apples-to-apples comparisons.

## Storyboard

The animation keeps a **roadmap pill-bar** at the top that lights up the
current step in yellow and greys out future steps, so viewers always know
where they are:

| #   | Step           | What happens                                                                                 |
| --- | -------------- | -------------------------------------------------------------------------------------------- |
| 1   | Problem        | Show A, B, empty C; state `c_ij = Σ a_ik · b_kj`.                                            |
| 2   | By hand        | Compute `c₀₀` in full, `c₀₁` compactly, then fill the rest left-to-right, top-to-bottom.     |
|     |                | Includes a "MAC = Multiply-Accumulate" definition callout.                                   |
| 3   | CPU            | One ALU + triple for-loop pseudocode + MAC counter ticking 1 → 27.                           |
| 4   | GPU (SIMT)     | 3×3 warp of threads; all 9 fire together AND each tile shows its own `a·b` and running acc. |
| 5   | Memory cost    | Every A row read 3×, every B col read 3× → **54 reads for 27 MACs**.                         |
| 6   | Systolic       | 3×3 PE grid, output-stationary. A flows →, B flows ↓. 7 cycles. **18 reads.**                |
| 7   | Trade-offs     | Side-by-side table (parallelism, memory, energy, flexibility, programming model).            |
| 8   | Recap          | Yellow "Recap" card, numbered bullets, tagline.                                              |

## Design notes on specific scenes

### Why all 9 GPU tiles flash together

SIMT = **Single Instruction, Multiple Threads**. By definition, every thread
in a warp executes the same instruction at the same clock cycle, just on
different data. That's the defining feature of a GPU warp. To show this
correctly the animation flashes all 9 tiles **in sync** — but each tile
also renders its own `a_ik · b_kj` expression and its own running
accumulator. The viewer sees both the parallelism (everyone active at
once) and the per-thread specialisation (different numbers in every tile)
in a single beat.

### Why the "by hand" panel lives at the bottom-left corner

In the first draft the working-out was anchored directly below the matrix
row, and the orange sum line (`1 + 0 + 6 = 7`) ended up hiding behind the
step-2 caption at the bottom of the screen. The panel is now pinned at a
fixed y-coordinate (`y ≈ -2.05`) with the product lines on the left and
the sum line to their right, so both stay safely above the caption strip
at the bottom edge.

## Why output-stationary for the systolic array?

The animation uses the **output-stationary** variant (each PE owns one
`c_ij`, A streams in from the left, B from the top, accumulator lives
locally). Reasons:

- It's what AWS Trainium's NeuronCore actually uses, so the analogy is
  accurate.
- For a 3×3 example it draws cleanly — the wavefront enters diagonally,
  peaks with all 9 PEs busy, then drains over 7 cycles total.
- The final `c_ij` lands in the PE at position `(i, j)`, which is where
  the viewer intuitively expects it.

The Google TPU uses a *weight-stationary* variant — functionally
equivalent for matmul, just a different choreography. A single caption
mentions this.

## LaTeX is used for math

All matrices, dot-product expressions, `Σ` notation, big-O, and the
"Cycle *N*" clocks are rendered via `MathTex` / `Matrix`. A working LaTeX
install is required — see **§7** of [`manim_setup.md`](manim_setup.md)
for the BasicTeX install instructions.

## Rendering

Activate the env first:

```bash
conda activate manim-env
```

Then from the repo root:

| Purpose                         | Command                                                                     |
| ------------------------------- | --------------------------------------------------------------------------- |
| Fast preview (~90 s, 480p15)    | `manim -pql manim_scripts/matmul_gpu_vs_systolic.py MatmulGPUvsSystolic`    |
| Teaching-grade (1080p60)        | `manim -pqh manim_scripts/matmul_gpu_vs_systolic.py MatmulGPUvsSystolic`    |
| 4K master                       | `manim -pqk manim_scripts/matmul_gpu_vs_systolic.py MatmulGPUvsSystolic`    |
| Animated GIF (shorter/lower-q)  | `manim --format=gif -ql manim_scripts/matmul_gpu_vs_systolic.py MatmulGPUvsSystolic` |

Output lands at:

```
media/videos/matmul_gpu_vs_systolic/<resolution>/MatmulGPUvsSystolic.mp4
```

A successful low-quality render (480p15) produced a ~3.6 MB file with
200 animations and no errors — that's the smoke test.

## Visual vocabulary

Kept consistent with [`llm_dataflow.py`](../manim_scripts/llm_dataflow.py)
so the two animations feel like siblings:

| Colour   | Meaning                                                           |
| -------- | ----------------------------------------------------------------- |
| BLUE     | `A` (left operand, "activations")                                 |
| GREEN    | `B` (right operand, "weights")                                    |
| ORANGE   | `C` / accumulators / partial sums                                 |
| YELLOW   | Currently-active row / column / PE / step                         |
| GREY     | Idle / dimmed / future step                                       |

## Tweaking

Common customisations, all near the top of `matmul_gpu_vs_systolic.py`:

- **The matrices**: change `A` and `B`. `C` is recomputed automatically.
  Keep them to small integers so the hand-computation is verifiable.
- **Matrix size**: `N = 3`. The systolic scene generalises (total cycles
  = `3N − 2`) but readability suffers fast past N=4.
- **Which cells to walk through in detail**: inside `_by_hand()`,
  `_walk_single_cell(0, 0, detailed=True)` runs the full blow-by-blow;
  the second call with `detailed=False` does a compact version; the
  `remaining` list batch-fills the rest.
- **CPU tick speed**: inside `_cpu_sequential()`, `run_time = 0.18 if
  mac_n <= 6 else 0.06` — bump the second value up if viewers complain
  the sweep is too fast.

## Intended audience & teaching notes

- Built for the **same IT-professional audience** as the LLM animation:
  comfortable with matrices, not necessarily with CUDA, warps, or
  dataflow architectures.
- Each sub-scene has a caption at the bottom so the video can be paused
  and read as static frames during a talk.
- The **18 vs. 54 reads** number is the emotional punchline — once a
  viewer sees that a systolic array reads each input value *exactly
  once*, the O(n³) memory-bandwidth problem of naive matmul on GPUs
  becomes tangible.
- The mention of Trainium's **128×128 grid** scales the toy up: the same
  choreography, running 16,384 MACs every single clock cycle, is why
  these chips exist.
- The closing tagline — *"less memory traffic = less energy = more
  FLOPs/watt"* — connects the architecture to the actual reason
  hyperscalers build custom silicon.
