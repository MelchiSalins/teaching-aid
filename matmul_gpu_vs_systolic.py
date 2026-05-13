"""Matrix-multiplication teaching animation.

Compares three ways to compute the SAME 3x3 matrix multiply:

    1. By hand (how a human does it on paper).
    2. CPU baseline (one scalar MAC at a time, 27 ticks).
    3. NVIDIA GPU / SIMT (a warp of threads, every output cell in parallel).
    4. Systolic array (AWS Trainium NeuronCore style, output-stationary).

The goal is to let an IT-savvy viewer follow the exact same toy computation
through each architecture and see WHY a systolic array is so much more
energy- and bandwidth-efficient than a GPU for a pure matmul.

Render from the project root (manim-env active, BasicTeX on PATH):

    manim -pql manim_scripts/matmul_gpu_vs_systolic.py MatmulGPUvsSystolic
    manim -pqh manim_scripts/matmul_gpu_vs_systolic.py MatmulGPUvsSystolic
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_E,
    DOWN,
    GREEN,
    GREEN_E,
    GREY,
    GREY_B,
    LEFT,
    ORANGE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Create,
    FadeIn,
    FadeOut,
    Indicate,
    Line,
    MathTex,
    Matrix,
    Rectangle,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
    Write,
)

# ---------------------------------------------------------------------------
# Colour vocabulary (intentionally matches llm_dataflow.py)
# ---------------------------------------------------------------------------
A_COLOR = BLUE          # left operand (activations)
B_COLOR = "#2ecc71"     # right operand (weights)
C_COLOR = ORANGE        # output / accumulator
HI_COLOR = YELLOW
IDLE_COLOR = GREY

# ---------------------------------------------------------------------------
# The worked example (verified by hand)
#
#   A @ B = C
#
#   [1 2 3]   [1 0 2]     [ 7  5 13]
#   [4 5 6] @ [0 1 1]  =  [16 11 31]
#   [7 8 9]   [2 1 3]     [25 17 49]
# ---------------------------------------------------------------------------
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
B = np.array([[1, 0, 2], [0, 1, 1], [2, 1, 3]], dtype=int)
C = A @ B  # [[7,5,13],[16,11,31],[25,17,49]]
N = 3      # all three matrices are N x N


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def mat_mob(values, color=WHITE, scale=0.75):
    entries = [[str(c) for c in row] for row in values]
    m = Matrix(entries, h_buff=0.85, v_buff=0.65, bracket_h_buff=0.15)
    m.set_color(color)
    m.scale(scale)
    return m


def pill(text: str, color=BLUE_E, width=1.9, height=0.55):
    box = Rectangle(width=width, height=height, color=color, fill_opacity=0.25)
    label = Text(text, font_size=20, color=WHITE)
    label.move_to(box.get_center())
    return VGroup(box, label)


def pe_cell(idx_label: str, side=1.05):
    """A processing element: a square with a label and an 'acc = 0' readout."""
    box = Rectangle(
        width=side, height=side, color=IDLE_COLOR, fill_opacity=0.05, stroke_width=2
    )
    idx = Text(idx_label, font_size=14, color=GREY_B)
    idx.move_to(box.get_corner(UP + LEFT) + RIGHT * 0.18 + DOWN * 0.13)
    acc = MathTex(r"0", color=C_COLOR).scale(0.65)
    acc.move_to(box.get_center())
    return VGroup(box, idx, acc)


# ---------------------------------------------------------------------------
# The main scene
# ---------------------------------------------------------------------------
class MatmulGPUvsSystolic(Scene):
    """GPU vs. systolic array: a 3x3 matmul, three ways."""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def construct(self) -> None:
        self._title_card()
        self.roadmap = self._build_roadmap()
        self.add(self.roadmap)

        self._highlight_step(0)
        self._problem_setup()

        self._highlight_step(1)
        self._by_hand()

        self._highlight_step(2)
        self._cpu_sequential()

        self._highlight_step(3)
        self._gpu_parallel()

        self._highlight_step(4)
        self._gpu_memory_cost()

        self._highlight_step(5)
        self._systolic_array()

        self._highlight_step(6)
        self._compare()

        self._summary()

    # ------------------------------------------------------------------
    # Title card
    # ------------------------------------------------------------------
    def _title_card(self) -> None:
        title = Text(
            "Matrix Multiplication: From Pencil to Silicon",
            font_size=48,
            color=YELLOW,
        )
        sub = Text(
            "CPU  vs.  GPU (NVIDIA SIMT)  vs.  Systolic Array (AWS Trainium)",
            font_size=26,
            color=BLUE,
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(title))
        self.play(FadeIn(sub, shift=DOWN))
        self.wait(1.4)
        self.play(FadeOut(title), FadeOut(sub))

    # ------------------------------------------------------------------
    # Roadmap pill-bar
    # ------------------------------------------------------------------
    def _build_roadmap(self) -> VGroup:
        steps_text = [
            "Problem",
            "By hand",
            "CPU",
            "GPU (SIMT)",
            "Memory cost",
            "Systolic",
            "Trade-offs",
        ]
        pills = VGroup(*[pill(s, color=GREY, width=1.7, height=0.5) for s in steps_text])
        pills.arrange(RIGHT, buff=0.15)
        pills.scale(0.95).to_edge(UP, buff=0.25)

        arrows = VGroup()
        for a, b in zip(pills[:-1], pills[1:]):
            arr = Arrow(
                a.get_right(),
                b.get_left(),
                buff=0.02,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(arr)
        self.steps = pills
        return VGroup(pills, arrows)

    def _highlight_step(self, idx: int) -> None:
        anims = []
        for i, p in enumerate(self.steps):
            box, label = p
            if i == idx:
                anims.append(
                    box.animate.set_fill(YELLOW, opacity=0.45).set_stroke(YELLOW)
                )
                anims.append(label.animate.set_color(WHITE))
            elif i < idx:
                anims.append(
                    box.animate.set_fill(GREEN_E, opacity=0.25).set_stroke(GREEN_E)
                )
                anims.append(label.animate.set_color(WHITE))
            else:
                anims.append(
                    box.animate.set_fill(GREY, opacity=0.15).set_stroke(GREY)
                )
                anims.append(label.animate.set_color(GREY_B))
        self.play(*anims, run_time=0.5)

    # ==================================================================
    # 1. Problem setup
    # ==================================================================
    def _problem_setup(self) -> None:
        caption = Text(
            "Step 1:  Our shared workload — a 3x3 matrix multiply.",
            font_size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))

        A_mat = mat_mob(A, color=A_COLOR, scale=0.85)
        B_mat = mat_mob(B, color=B_COLOR, scale=0.85)
        C_empty = mat_mob(
            [["?", "?", "?"], ["?", "?", "?"], ["?", "?", "?"]],
            color=GREY_B,
            scale=0.85,
        )

        # Put the equation well above the matrices so it doesn't collide
        # with the A / B / C labels that sit above each matrix bracket.
        eq1 = MathTex("C", "=", "A", r"\cdot", "B", font_size=44)
        eq1[0].set_color(C_COLOR)
        eq1[2].set_color(A_COLOR)
        eq1[4].set_color(B_COLOR)
        eq1.move_to(UP * 2.4)

        self.play(Write(eq1))
        self.wait(0.4)

        A_label = MathTex("A", color=A_COLOR).scale(0.9)
        B_label = MathTex("B", color=B_COLOR).scale(0.9)
        C_label = MathTex("C", color=C_COLOR).scale(0.9)

        row = VGroup(A_mat, MathTex(r"\cdot").scale(1.1),
                     B_mat, MathTex("=").scale(1.1), C_empty)
        row.arrange(RIGHT, buff=0.35)
        row.shift(DOWN * 0.5)

        A_label.next_to(A_mat, UP, buff=0.15)
        B_label.next_to(B_mat, UP, buff=0.15)
        C_label.next_to(C_empty, UP, buff=0.15)

        self.play(
            FadeIn(row),
            FadeIn(A_label), FadeIn(B_label), FadeIn(C_label),
        )
        self.wait(0.8)

        rule = MathTex(
            r"c_{ij} \;=\; \sum_{k=0}^{2} a_{ik}\, b_{kj}",
            font_size=34,
        ).next_to(row, DOWN, buff=0.45)
        rule.set_color_by_tex("c_{ij}", C_COLOR)
        self.play(Write(rule))
        self.wait(1.4)

        self.play(
            FadeOut(eq1), FadeOut(rule),
            FadeOut(A_label), FadeOut(B_label), FadeOut(C_label),
        )
        # Keep the three matrices around for the "by hand" pass.
        self.A_mat = A_mat
        self.B_mat = B_mat
        self.C_mat = C_empty
        self.row_group = row
        self.play(FadeOut(caption))

    # ==================================================================
    # 2. By hand
    # ==================================================================
    def _by_hand(self) -> None:
        caption = Text(
            "Step 2:  First, let's do it the way you learned in school.",
            font_size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))

        # Detailed walk-through of c_00 = 1*1 + 2*0 + 3*2 = 7
        # (top-left cell — where a human would start)
        self._walk_single_cell(0, 0, detailed=True)

        # Introduce MAC terminology on-screen the first time we say
        # "multiply-adds". Placed above the caption, away from the panel.
        mac_def = MathTex(
            r"\text{MAC} = \text{Multiply-Accumulate}:\ "
            r"\mathrm{acc} \mathrel{+}= a \times b",
            font_size=28,
            color=HI_COLOR,
        ).next_to(caption, UP, buff=0.2)
        self.play(FadeIn(mac_def))
        self.wait(1.6)
        self.play(FadeOut(mac_def))

        # Quicker walk-through of c_01 = 1*0 + 2*1 + 3*1 = 5
        # (next cell to the right — natural reading order)
        self._walk_single_cell(0, 1, detailed=False)

        # Fast-forward: fill the remaining cells in strict reading order
        # (top row → middle row → bottom row, left → right)
        remaining = [
            (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]
        for (i, j) in remaining:
            self._fill_cell(i, j)

        # Final summary: 27 MACs
        mac_note = MathTex(
            r"27\ \text{MACs total}\ \text{—}\ \text{done by one brain,\ sequentially.}",
            font_size=30,
            color=HI_COLOR,
        ).next_to(caption, UP, buff=0.2)
        self.play(FadeIn(mac_note))
        self.wait(1.6)
        self.play(FadeOut(mac_note), FadeOut(caption))

    def _walk_single_cell(self, i: int, j: int, detailed: bool) -> None:
        """Animate the dot-product that fills C[i,j]."""
        # Highlight row i of A and column j of B
        A_rows = self.A_mat.get_rows()
        B_cols = self.B_mat.get_columns()

        row_box = SurroundingRectangle(A_rows[i], color=A_COLOR, buff=0.08)
        col_box = SurroundingRectangle(B_cols[j], color=B_COLOR, buff=0.08)
        self.play(Create(row_box), Create(col_box), run_time=0.6)

        # Computation panel below
        products = []
        for k in range(N):
            a_val = int(A[i, k])
            b_val = int(B[k, j])
            prod = a_val * b_val
            products.append((a_val, b_val, prod))

        if detailed:
            # Long form: show each product, then the sum.
            # Place the working-out panel BELOW and to the LEFT of the row
            # group at a fixed y-position, well above the caption, so the
            # orange sum line is never hidden by the caption strip.
            lines = VGroup()
            sum_terms = []
            for k, (a_val, b_val, prod) in enumerate(products):
                line = MathTex(
                    rf"{a_val}", r"\cdot", rf"{b_val}", "=", rf"{prod}",
                    font_size=30,
                )
                line[0].set_color(A_COLOR)
                line[2].set_color(B_COLOR)
                line[4].set_color(C_COLOR)
                lines.add(line)
                sum_terms.append(prod)
            lines.arrange(DOWN, aligned_edge=LEFT, buff=0.14)

            # Fixed anchor: left edge of screen, centred vertically around
            # y ≈ -2.1 (matrices sit around y 0 .. -1). This puts the three
            # product lines and the sum line well above the caption at
            # y ≈ -3.5 — no more overlap.
            panel_anchor = np.array([-5.6, -2.05, 0])
            lines.move_to(panel_anchor, aligned_edge=LEFT)

            for k, line in enumerate(lines):
                # Tiny arrows showing which a_ik and b_kj we're grabbing
                a_entry = A_rows[i][k]
                b_entry = B_cols[j][k]
                a_ring = SurroundingRectangle(a_entry, color=HI_COLOR, buff=0.05)
                b_ring = SurroundingRectangle(b_entry, color=HI_COLOR, buff=0.05)
                self.play(Create(a_ring), Create(b_ring), run_time=0.35)
                self.play(Write(line), run_time=0.55)
                self.play(FadeOut(a_ring), FadeOut(b_ring), run_time=0.25)

            total = sum(sum_terms)
            sum_line = MathTex(
                "+".join(str(t) for t in sum_terms), "=", rf"{total}",
                font_size=32,
                color=C_COLOR,
            )
            # Anchor the sum line to the RIGHT of the product lines (not
            # below them) so it stays at the same y as the products and
            # can't collide with the bottom caption.
            sum_line.next_to(lines, RIGHT, buff=0.55, aligned_edge=DOWN)
            self.play(Write(sum_line))
            self.wait(0.6)

            # Drop the result into C[i,j]
            target = self.C_mat.get_entries()[i * N + j]
            new_val = MathTex(str(int(C[i, j])), color=C_COLOR).scale(0.85)
            new_val.move_to(target.get_center())
            self.play(
                Transform(target, new_val),
                FadeOut(lines), FadeOut(sum_line),
                run_time=0.9,
            )
            self.play(FadeOut(row_box), FadeOut(col_box), run_time=0.3)
        else:
            # Compact form: single line showing the whole dot product,
            # anchored at the LEFT edge to keep it well away from the caption.
            a_vals = [p[0] for p in products]
            b_vals = [p[1] for p in products]
            total = sum(p[2] for p in products)
            line = MathTex(
                rf"{a_vals[0]}\cdot{b_vals[0]}",
                "+",
                rf"{a_vals[1]}\cdot{b_vals[1]}",
                "+",
                rf"{a_vals[2]}\cdot{b_vals[2]}",
                "=",
                rf"{total}",
                font_size=32,
            )
            line.set_color(WHITE)
            line[-1].set_color(C_COLOR)
            # Same fixed y anchor as the detailed form — above the caption,
            # below the matrices.
            line.move_to(np.array([-3.2, -2.05, 0]))
            self.play(Write(line), run_time=0.8)
            self.wait(0.5)

            target = self.C_mat.get_entries()[i * N + j]
            new_val = MathTex(str(int(C[i, j])), color=C_COLOR).scale(0.85)
            new_val.move_to(target.get_center())
            self.play(
                Transform(target, new_val),
                FadeOut(line),
                FadeOut(row_box), FadeOut(col_box),
                run_time=0.7,
            )

    def _fill_cell(self, i: int, j: int) -> None:
        """Quick fill — flash row/col and drop the answer in."""
        A_rows = self.A_mat.get_rows()
        B_cols = self.B_mat.get_columns()
        row_box = SurroundingRectangle(A_rows[i], color=A_COLOR, buff=0.06)
        col_box = SurroundingRectangle(B_cols[j], color=B_COLOR, buff=0.06)

        target = self.C_mat.get_entries()[i * N + j]
        new_val = MathTex(str(int(C[i, j])), color=C_COLOR).scale(0.85)
        new_val.move_to(target.get_center())

        self.play(
            Create(row_box), Create(col_box),
            Transform(target, new_val),
            run_time=0.35,
        )
        self.play(FadeOut(row_box), FadeOut(col_box), run_time=0.15)

    # ==================================================================
    # 3. CPU baseline — one ALU, triple nested loop
    # ==================================================================
    def _cpu_sequential(self) -> None:
        caption = Text(
            "Step 3:  A CPU does it the same way — one MAC per clock tick.",
            font_size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))

        # Shrink and move the matrices to the left. We keep the row_group
        # reference because A_mat / B_mat / C_mat still live inside it.
        self.play(
            self.row_group.animate.scale(0.7).to_edge(LEFT, buff=0.4).shift(UP * 0.2)
        )

        # The C matrix currently shows the fully-computed values from the
        # "by hand" scene. Clear it back to "?" so the CPU can re-produce
        # each cell itself — otherwise viewers see the answer before the
        # CPU actually does the work.
        c_entries = self.C_mat.get_entries()
        blank_anims = []
        blanks = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                entry = c_entries[i * N + j]
                blank = MathTex("?", color=GREY_B).scale(0.85 * 0.7).move_to(
                    entry.get_center()
                )
                blanks[i][j] = blank
                blank_anims.append(Transform(entry, blank))
        self.play(*blank_anims, run_time=0.5)

        # A single ALU block on the right
        alu = Rectangle(width=2.4, height=1.2, color=WHITE, fill_opacity=0.08)
        alu_label = Text("CPU core\n(1 ALU)", font_size=20, color=WHITE)
        alu_label.move_to(alu.get_center())
        alu_group = VGroup(alu, alu_label)
        alu_group.shift(RIGHT * 3.4 + UP * 1.6)

        # Loop pseudocode
        code = VGroup(
            Text("for i in 0..2:", font_size=22, color=WHITE),
            Text("  for j in 0..2:", font_size=22, color=WHITE),
            Text("    c[i][j] = 0", font_size=22, color=WHITE),
            Text("    for k in 0..2:", font_size=22, color=WHITE),
            Text("      c[i][j] += a[i][k] * b[k][j]",
                 font_size=22, color=HI_COLOR),
        )
        code.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        code.next_to(alu_group, DOWN, buff=0.3)

        # MAC counter
        mac_counter = MathTex(r"\text{MACs:}\; 0", font_size=28, color=HI_COLOR)
        mac_counter.to_edge(RIGHT, buff=0.5).shift(UP * 2.8)

        # Live "this MAC" readout — sits next to the ALU and tells the
        # viewer exactly which element of C is currently being updated.
        current_op = MathTex("", font_size=26)
        current_op.next_to(alu_group, LEFT, buff=0.4)

        # Index indicator (i, j, k = …) underneath the pseudocode so the
        # viewer can see which iteration of the triple loop is running.
        idx_readout = MathTex(
            r"i=0,\ j=0,\ k=0", font_size=22, color=GREY_B,
        )
        idx_readout.next_to(code, DOWN, buff=0.3)

        self.play(
            FadeIn(alu_group), FadeIn(code),
            FadeIn(mac_counter), FadeIn(idx_readout),
        )
        self.wait(0.4)

        # Track the running partial sum for each output cell. The viewer
        # will see these tick from 0 → partial → final value as each
        # inner-loop MAC happens.
        partial = np.zeros((N, N), dtype=int)

        # Entries helper refs (updated via Transform as the scene runs)
        A_entries = self.A_mat.get_entries()
        B_entries = self.B_mat.get_entries()

        mac_n = 0
        # We'll keep a reference to the *current* C cell mobject so we
        # can Transform it in-place each MAC.
        cur_c = [[c_entries[i * N + j] for j in range(N)] for i in range(N)]

        for i in range(N):
            for j in range(N):
                # Yellow ring around the C cell we're about to populate —
                # stays visible for all 3 MACs that contribute to it.
                c_cell = cur_c[i][j]
                c_ring = SurroundingRectangle(c_cell, color=HI_COLOR, buff=0.05)
                self.play(Create(c_ring), run_time=0.2)

                for k in range(N):
                    mac_n += 1
                    # Pick pacing by position in the schedule
                    if mac_n <= 3:          # first cell (c_00): teach slowly
                        rt = 0.55
                        show_full_op = True
                    elif mac_n <= 9:        # next two cells: medium
                        rt = 0.25
                        show_full_op = True
                    else:                   # rest: fast sweep
                        rt = 0.1
                        show_full_op = False

                    a_val = int(A[i, k])
                    b_val = int(B[k, j])
                    partial[i, j] += a_val * b_val

                    # Highlight the A and B elements being read this cycle
                    a_entry = A_entries[i * N + k]
                    b_entry = B_entries[k * N + j]
                    a_ring = SurroundingRectangle(a_entry, color=A_COLOR, buff=0.04)
                    b_ring = SurroundingRectangle(b_entry, color=B_COLOR, buff=0.04)

                    # Update index readout
                    new_idx = MathTex(
                        rf"i={i},\ j={j},\ k={k}",
                        font_size=22, color=GREY_B,
                    ).move_to(idx_readout.get_center())

                    # Update MAC counter
                    new_counter = MathTex(
                        rf"\text{{MACs:}}\; {mac_n}\ /\ 27",
                        font_size=28, color=HI_COLOR,
                    ).move_to(mac_counter.get_center())

                    # Build the "what's happening" text near the ALU
                    if show_full_op:
                        new_op = MathTex(
                            rf"c[{i}][{j}]\mathrel{{+}}={a_val}\cdot{b_val}",
                            font_size=24, color=HI_COLOR,
                        ).next_to(alu_group, LEFT, buff=0.4)
                    else:
                        new_op = MathTex(
                            rf"c[{i}][{j}]", font_size=24, color=HI_COLOR,
                        ).next_to(alu_group, LEFT, buff=0.4)

                    # Update the C cell to show the running partial sum.
                    # Colour stays yellow while it's mid-accumulation, flips
                    # to orange when the final MAC of this cell completes.
                    is_last = (k == N - 1)
                    new_c_val = MathTex(
                        str(int(partial[i, j])),
                        color=(C_COLOR if is_last else HI_COLOR),
                    ).scale(0.85 * 0.7).move_to(c_cell.get_center())

                    self.play(
                        Create(a_ring), Create(b_ring),
                        Indicate(alu, color=HI_COLOR, scale_factor=1.08),
                        Transform(mac_counter, new_counter),
                        Transform(idx_readout, new_idx),
                        Transform(current_op, new_op),
                        Transform(c_cell, new_c_val),
                        run_time=rt,
                    )
                    # Quickly clear the A/B surrounds so they don't pile up
                    self.play(
                        FadeOut(a_ring), FadeOut(b_ring),
                        run_time=max(0.05, rt * 0.3),
                    )

                # Done with this c_ij — drop the yellow cell ring
                self.play(FadeOut(c_ring), run_time=0.15)

        # Clear the lingering "current op" line
        self.play(FadeOut(current_op), run_time=0.2)

        self.wait(0.3)
        punch = MathTex(
            r"27\ \text{MACs}\ \Rightarrow\ 27\ \text{clock cycles.}",
            font_size=30,
            color=HI_COLOR,
        ).next_to(idx_readout, DOWN, buff=0.3)
        self.play(Write(punch))
        self.wait(1.6)

        self.play(
            FadeOut(alu_group), FadeOut(code),
            FadeOut(mac_counter), FadeOut(idx_readout),
            FadeOut(punch), FadeOut(caption),
        )
        # Keep matrices on screen for GPU/systolic phases

    # ==================================================================
    # 4. GPU / SIMT — 9 threads, 1 per output cell
    # ==================================================================
    def _gpu_parallel(self) -> None:
        caption = Text(
            "Step 4:  A GPU warp fires many threads — one per output cell.",
            font_size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))

        # Slide the matrices back to centre-left and clear C
        self.play(
            self.row_group.animate.scale(1.0).move_to(LEFT * 3.4 + UP * 0.3)
        )
        # Reset C to "?"
        for i in range(N):
            for j in range(N):
                entry = self.C_mat.get_entries()[i * N + j]
                blank = MathTex("?", color=GREY_B).scale(0.85).move_to(
                    entry.get_center()
                )
                self.add(blank)
                self.remove(entry)
                # Replace reference so future updates use the new mobject
                self.C_mat.get_entries().submobjects[i * N + j] = blank

        # Draw a 3x3 grid of "threads" on the right.
        # Each thread tile is larger now because it also has to display
        # the thread name (T_ij), the live "a·b" operation it's computing
        # this cycle, and its running accumulator.
        thread_side = 1.15
        thread_grid = VGroup()
        thread_boxes = [[None] * N for _ in range(N)]
        thread_names = [[None] * N for _ in range(N)]
        thread_ops  = [[None] * N for _ in range(N)]   # shows "a·b"
        thread_accs = [[None] * N for _ in range(N)]   # shows running acc
        for i in range(N):
            row_group = VGroup()
            for j in range(N):
                box = Rectangle(
                    width=thread_side, height=thread_side,
                    color=IDLE_COLOR, fill_opacity=0.08, stroke_width=2,
                )
                name = Text(f"T{i}{j}", font_size=13, color=GREY_B)
                name.move_to(box.get_corner(UP + LEFT) + RIGHT * 0.22 + DOWN * 0.17)
                op = MathTex("", font_size=24).move_to(
                    box.get_center() + UP * 0.18
                )
                acc = MathTex("0", color=C_COLOR, font_size=26).move_to(
                    box.get_center() + DOWN * 0.25
                )
                cell = VGroup(box, name, op, acc)
                thread_boxes[i][j] = box
                thread_names[i][j] = name
                thread_ops[i][j] = op
                thread_accs[i][j] = acc
                row_group.add(cell)
            row_group.arrange(RIGHT, buff=0.12)
            thread_grid.add(row_group)
        thread_grid.arrange(DOWN, buff=0.12)
        thread_grid.shift(RIGHT * 3.4 + UP * 0.3)

        warp_label = Text(
            "Warp: 9 threads  (real warps = 32 lanes)",
            font_size=20, color=WHITE,
        ).next_to(thread_grid, UP, buff=0.25)

        self.play(FadeIn(thread_grid), FadeIn(warp_label))
        self.wait(0.5)

        # Introduce SIMT with a single gentle flash of all 9 tiles — this
        # is the "every thread owns one output cell" beat.
        flashes = []
        for i in range(N):
            for j in range(N):
                flashes.append(
                    thread_boxes[i][j].animate.set_fill(HI_COLOR, opacity=0.25)
                    .set_stroke(HI_COLOR)
                )
        self.play(*flashes, run_time=0.6)

        note1 = Text(
            "All 9 threads execute the SAME instruction (SIMT) — on DIFFERENT data.",
            font_size=20, color=GREY_B,
        ).next_to(warp_label, UP, buff=0.15)
        self.play(FadeIn(note1))
        self.wait(0.6)

        # Clock counter (renamed "MAC step" so viewers still connect it
        # to the MAC concept introduced in the by-hand scene).
        mac_counter = MathTex(r"\text{MAC step:}\; 0", font_size=26, color=HI_COLOR)
        mac_counter.next_to(thread_grid, DOWN, buff=0.3)
        self.play(FadeIn(mac_counter))

        # Track each thread's running accumulator so the viewer can watch
        # it climb to the final c_ij over 3 cycles.
        acc_vals = np.zeros((N, N), dtype=int)

        # Step through the 3 MAC cycles. Each cycle, EVERY thread reads
        # its own a[i][k] and b[k][j] and performs one MAC.
        for k in range(N):
            new_counter = MathTex(
                rf"\text{{MAC step:}}\; {k + 1}\ /\ 3",
                font_size=26, color=HI_COLOR,
            ).move_to(mac_counter.get_center())

            # Highlight the column of A and the row of B that ALL threads
            # are reading from this cycle.
            A_col = VGroup(*[self.A_mat.get_entries()[i * N + k] for i in range(N)])
            B_row = VGroup(*[self.B_mat.get_entries()[k * N + j] for j in range(N)])
            col_box = SurroundingRectangle(A_col, color=A_COLOR, buff=0.05)
            row_box = SurroundingRectangle(B_row, color=B_COLOR, buff=0.05)

            self.play(
                Create(col_box), Create(row_box),
                Transform(mac_counter, new_counter),
                run_time=0.45,
            )

            # Build per-tile "a·b" expressions — all 9 appear simultaneously.
            op_transforms = []
            pulses = []
            for i in range(N):
                for j in range(N):
                    a_val = int(A[i, k])
                    b_val = int(B[k, j])
                    new_op = MathTex(
                        rf"{a_val}", r"\cdot", rf"{b_val}",
                        font_size=26,
                    )
                    new_op[0].set_color(A_COLOR)
                    new_op[2].set_color(B_COLOR)
                    new_op.move_to(thread_ops[i][j].get_center())
                    op_transforms.append(Transform(thread_ops[i][j], new_op))
                    pulses.append(
                        thread_boxes[i][j].animate.set_fill(HI_COLOR, opacity=0.55)
                        .set_stroke(HI_COLOR)
                    )
            # All 9 threads pulse together AND each shows its own data.
            self.play(*pulses, *op_transforms, run_time=0.55)

            # Update each thread's running accumulator simultaneously.
            acc_transforms = []
            for i in range(N):
                for j in range(N):
                    acc_vals[i, j] += int(A[i, k]) * int(B[k, j])
                    new_acc = MathTex(
                        rf"\mathrm{{acc}}={int(acc_vals[i, j])}",
                        color=C_COLOR, font_size=24,
                    ).move_to(thread_accs[i][j].get_center())
                    acc_transforms.append(Transform(thread_accs[i][j], new_acc))
            self.play(*acc_transforms, run_time=0.5)

            # Cool the tiles back to "active but not mid-MAC" colour
            reset = [
                thread_boxes[i][j].animate.set_fill(HI_COLOR, opacity=0.18)
                .set_stroke(HI_COLOR)
                for i in range(N) for j in range(N)
            ]
            self.play(*reset, FadeOut(col_box), FadeOut(row_box), run_time=0.3)

        # Each thread now holds its final c_ij in its accumulator — copy
        # those values into the C matrix on the left.
        writes = []
        for i in range(N):
            for j in range(N):
                target = self.C_mat.get_entries()[i * N + j]
                new_val = MathTex(str(int(C[i, j])), color=C_COLOR).scale(0.85)
                new_val.move_to(target.get_center())
                writes.append(Transform(target, new_val))
        self.play(*writes, run_time=0.8)
        self.wait(0.4)

        # Clear the per-tile op text now that we're done (keeps the frame
        # cleaner for the memory-cost scene that follows).
        self.play(
            *[FadeOut(thread_ops[i][j]) for i in range(N) for j in range(N)],
            run_time=0.3,
        )

        punch = MathTex(
            r"27\ \text{MACs}\ \Rightarrow\ 3\ \text{cycles}\ "
            r"(9\times\ \text{parallel})",
            font_size=28, color=HI_COLOR,
        ).next_to(caption, UP, buff=0.2)
        self.play(Write(punch))
        self.wait(1.6)

        self.play(
            FadeOut(punch), FadeOut(note1), FadeOut(warp_label),
            FadeOut(mac_counter),
        )
        # Keep thread_grid for the memory-cost scene
        self.thread_grid = thread_grid
        self.play(FadeOut(caption))

    # ==================================================================
    # 5. GPU memory cost — 54 reads for 27 MACs
    # ==================================================================
    def _gpu_memory_cost(self) -> None:
        caption = Text(
            "Step 5:  ...but every thread pulls its own data from memory.",
            font_size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))

        # Focus on row 0 of A: threads T00, T01, T02 all read it
        A_rows = self.A_mat.get_rows()
        row0_box = SurroundingRectangle(A_rows[0], color=A_COLOR, buff=0.08)
        self.play(Create(row0_box))

        focus_threads = VGroup(
            self.thread_grid[0][0], self.thread_grid[0][1], self.thread_grid[0][2],
        )
        rings = VGroup(
            *[SurroundingRectangle(t, color=A_COLOR, buff=0.06) for t in focus_threads]
        )
        # Three arrows from row 0 of A to the three threads in row 0 of the grid
        arrows = VGroup()
        for t in focus_threads:
            arr = Arrow(
                row0_box.get_right(), t.get_left(),
                buff=0.1, stroke_width=2,
                color=A_COLOR,
                max_tip_length_to_length_ratio=0.1,
            )
            arrows.add(arr)
        self.play(Create(rings), *[Create(a) for a in arrows], run_time=0.9)

        dup_note = Text(
            "Row 0 of A is read 3 times  (once per consuming thread).",
            font_size=22, color=A_COLOR,
        ).next_to(caption, UP, buff=0.2)
        self.play(FadeIn(dup_note))
        self.wait(1.3)

        self.play(FadeOut(arrows), FadeOut(rings), FadeOut(row0_box),
                  FadeOut(dup_note))

        # Tally it up
        tally = VGroup(
            MathTex(
                r"\text{Reads of A}:\; 3\ \text{rows}\times 3\ \text{threads}"
                r"\times 3\ \text{cycles} = 27",
                font_size=26, color=A_COLOR,
            ),
            MathTex(
                r"\text{Reads of B}:\; 3\ \text{cols}\times 3\ \text{threads}"
                r"\times 3\ \text{cycles} = 27",
                font_size=26, color=B_COLOR,
            ),
            MathTex(
                r"\text{Total memory traffic}:\; 54\ \text{reads for }27\ "
                r"\text{MACs.}",
                font_size=28, color=HI_COLOR,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        tally.next_to(caption, UP, buff=0.3)
        self.play(FadeIn(tally, shift=UP * 0.2))
        self.wait(1.6)

        teaser = Text(
            "What if every input value was read only ONCE?",
            font_size=24, color=HI_COLOR,
        ).next_to(tally, UP, buff=0.3)
        self.play(Write(teaser))
        self.wait(1.5)

        self.play(
            FadeOut(self.thread_grid), FadeOut(self.row_group),
            FadeOut(tally), FadeOut(teaser), FadeOut(caption),
        )

    # ==================================================================
    # 6. Systolic array (output-stationary, 3x3)
    # ==================================================================
    def _systolic_array(self) -> None:
        caption = Text(
            "Step 6:  A systolic array — data FLOWS between neighbouring PEs.",
            font_size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))

        # ---------- 6a. Reference A and B matrices (top-left) ----------
        # The previous scene faded the big matrices; rebuild small reference
        # copies at the top-left so viewers can look up which numbers came
        # from where while chips fly around.
        A_ref = mat_mob(A, color=A_COLOR, scale=0.50)
        A_ref_lbl = MathTex("A", color=A_COLOR).scale(0.7)
        B_ref = mat_mob(B, color=B_COLOR, scale=0.50)
        B_ref_lbl = MathTex("B", color=B_COLOR).scale(0.7)

        ref_group = VGroup(
            VGroup(A_ref_lbl, A_ref).arrange(DOWN, buff=0.1),
            VGroup(B_ref_lbl, B_ref).arrange(DOWN, buff=0.1),
        ).arrange(RIGHT, buff=0.4)
        ref_group.to_corner(UP + LEFT, buff=0.5).shift(DOWN * 0.35)
        self.play(FadeIn(ref_group))
        self.wait(0.3)

        # ---------- 6b. Build the 3x3 PE grid ----------
        pe_side = 0.85
        spacing = 1.0
        pes = [[None] * N for _ in range(N)]
        pe_group = VGroup()
        for i in range(N):
            for j in range(N):
                cell = pe_cell(f"PE[{i},{j}]", side=pe_side)
                cell.move_to(np.array([j * spacing, -i * spacing, 0]))
                pes[i][j] = cell
                pe_group.add(cell)
        pe_group.move_to(np.array([1.8, -1.3, 0]))
        self.play(FadeIn(pe_group))

        # Clock label (top-right)
        clock = MathTex(r"\text{Cycle } 0", font_size=26, color=HI_COLOR)
        clock.to_corner(UP + RIGHT, buff=0.7).shift(DOWN * 0.3)
        self.play(FadeIn(clock))

        # ---------- 6c. Staggered A and B feeds ----------
        # Output-stationary systolic array: PE[i,j] needs a[i,k] · b[k,j]
        # at cycle (i + j + k + 1)  [1-indexed display].  So chip A[i,k]
        # must be at PE[i,0] at cycle (i + k + 1) and shift one PE to the
        # right each subsequent cycle.  The pre-staged starting position
        # (at cycle 0, before the clock ticks) is therefore
        #     PE[i,0].center  +  LEFT * (i + k + 1) * spacing
        # which for all (i,k) is a PARALLELOGRAM tipped to the bottom-left.
        # The B feed is the same pattern rotated 90° (top, tipped up-right).

        chip_size = 0.42
        def make_chip(val, colour):
            r = Rectangle(
                width=chip_size, height=chip_size,
                color=colour, fill_opacity=0.75, stroke_width=1.5,
            )
            t = MathTex(str(int(val)), color=WHITE).scale(0.5)
            t.move_to(r.get_center())
            return VGroup(r, t)

        # Build all 9 A-chips (one per A[i,k]) at their cycle-0 positions
        a_chips = {}
        for i in range(N):
            for k in range(N):
                c = make_chip(A[i, k], A_COLOR)
                c.move_to(
                    pes[i][0].get_center() + LEFT * (i + k + 1) * spacing
                )
                a_chips[(i, k)] = c

        # Build all 9 B-chips at their cycle-0 positions (above the grid)
        b_chips = {}
        for kk in range(N):
            for j in range(N):
                c = make_chip(B[kk, j], B_COLOR)
                c.move_to(
                    pes[0][j].get_center() + UP * (j + kk + 1) * spacing
                )
                b_chips[(kk, j)] = c

        # Staggered-feed intro caption
        stagger_note = Text(
            "A and B are fed in as STAGGERED streams — each row/col "
            "delayed by its index.",
            font_size=20, color=HI_COLOR,
        ).next_to(caption, UP, buff=0.15)
        self.play(FadeIn(stagger_note))

        # Introduce the feeds: fade them in row-by-row / col-by-col so the
        # staircase shape is obvious.
        for i in range(N):
            row_anims = [FadeIn(a_chips[(i, k)], shift=RIGHT * 0.3)
                         for k in range(N)]
            self.play(*row_anims, run_time=0.35)
        for j in range(N):
            col_anims = [FadeIn(b_chips[(kk, j)], shift=DOWN * 0.3)
                         for kk in range(N)]
            self.play(*col_anims, run_time=0.35)
        self.wait(0.8)

        # Explain the mechanic with a small caption under the pre-stage
        hint = Text(
            "Each cycle: A slides →, B slides ↓. When a chip lands in a PE,"
            " the PE MACs into its accumulator.",
            font_size=18, color=GREY_B,
        ).next_to(stagger_note, UP, buff=0.1)
        self.play(FadeIn(hint))
        self.wait(0.6)
        # Shrink the intro caption out once the viewer has read it
        self.play(FadeOut(stagger_note))

        # ---------- 6d. Run the 7 cycles ----------
        acc = np.zeros((N, N), dtype=int)
        total_cycles = 3 * N - 2  # = 7 for N=3 (last MAC is at PE[2,2])

        for t in range(1, total_cycles + 1):
            # Move every A chip one step right; every B chip one step down.
            move_anims = []
            for chip in a_chips.values():
                move_anims.append(chip.animate.shift(RIGHT * spacing))
            for chip in b_chips.values():
                move_anims.append(chip.animate.shift(DOWN * spacing))

            # Which PEs MAC this cycle?  PE[i,j] uses A[i,k] · B[k,j] where
            # k = t - 1 - i - j  (valid when 0 <= k < N).
            pe_pulses = []
            acc_transforms = []
            mac_list = []  # (i, j, k) for cells that MAC this cycle
            for i in range(N):
                for j in range(N):
                    k = t - 1 - i - j
                    if 0 <= k < N:
                        mac_list.append((i, j, k))
                        acc[i, j] += int(A[i, k]) * int(B[k, j])
                        old_acc = pes[i][j][2]
                        new_acc = MathTex(
                            str(int(acc[i, j])),
                            color=(C_COLOR if k == N - 1 else HI_COLOR),
                        ).scale(0.7).move_to(old_acc.get_center())
                        acc_transforms.append(Transform(old_acc, new_acc))
                        pe_pulses.append(
                            pes[i][j][0].animate.set_fill(HI_COLOR, opacity=0.45)
                            .set_stroke(HI_COLOR)
                        )

            # Update clock
            new_clock = MathTex(
                rf"\text{{Cycle }} {t}", font_size=26, color=HI_COLOR,
            ).move_to(clock.get_center())

            # Play movement + PE pulses together; accumulator updates right after.
            self.play(*move_anims, Transform(clock, new_clock),
                      *pe_pulses, run_time=0.6)
            if acc_transforms:
                self.play(*acc_transforms, run_time=0.4)

            # Cool PEs that MAC'd back down (but stay slightly warm)
            if pe_pulses:
                cool = [
                    pes[i][j][0].animate.set_fill(HI_COLOR, opacity=0.18)
                    .set_stroke(HI_COLOR)
                    for (i, j, _) in mac_list
                ]
                self.play(*cool, run_time=0.2)

            # Fade any chips that have now passed PE[i,N-1] (exited the grid).
            # A chip exits after cycle (i + k + N).
            exit_fades = []
            for (i, k) in list(a_chips.keys()):
                if t >= i + k + N:
                    exit_fades.append(FadeOut(a_chips[(i, k)]))
                    del a_chips[(i, k)]
            for (kk, j) in list(b_chips.keys()):
                if t >= kk + j + N:
                    exit_fades.append(FadeOut(b_chips[(kk, j)]))
                    del b_chips[(kk, j)]
            if exit_fades:
                self.play(*exit_fades, run_time=0.2)

        # ---------- 6e. Done — every PE now holds the correct c_ij ----------
        self.wait(0.6)

        # Flip every PE fill to orange to signal "this is C"
        final_anims = [
            pes[i][j][0].animate.set_fill(C_COLOR, opacity=0.25)
            .set_stroke(C_COLOR)
            for i in range(N) for j in range(N)
        ]
        self.play(*final_anims, run_time=0.6)
        self.wait(0.4)

        # ---------- 6f. Clean up before the closing beat ----------
        # Fade every element we don't need any more so the final summary
        # has a clean frame to live in.  The ref matrices top-left, the
        # grey "each cycle" hint, the clock, and any leftover edge chips
        # all go.  The caption at the bottom stays as a breadcrumb.
        leftover_chips = list(a_chips.values()) + list(b_chips.values())
        cleanup = [
            FadeOut(ref_group),
            FadeOut(hint),
            FadeOut(clock),
        ] + [FadeOut(c) for c in leftover_chips]
        self.play(*cleanup, run_time=0.5)
        # Dictionaries are now stale — clear them so we don't try to fade
        # these mobjects a second time at the end of the scene.
        a_chips.clear()
        b_chips.clear()

        # ---------- 6g. Reframe the PE grid as the result matrix C ----------
        # Move the whole PE grid to the LEFT half of the screen and pin a
        # yellow title above it so viewers read it as "C = A · B".  This
        # frees up the right half for the two yellow summary text blocks.
        self.play(
            pe_group.animate.scale(0.9).move_to(np.array([-3.2, -0.3, 0])),
            run_time=0.7,
        )
        result_title = MathTex(
            "C", "=", "A", r"\cdot", "B",
            font_size=40,
        )
        result_title[0].set_color(C_COLOR)
        result_title[2].set_color(A_COLOR)
        result_title[4].set_color(B_COLOR)
        result_title.next_to(pe_group, UP, buff=0.4)
        self.play(Write(result_title))
        self.wait(0.3)

        # ---------- 6h. Closing messages (right half of screen) ----------
        reads = MathTex(
            r"\text{Reads: } 9\ \text{from }A + 9\ \text{from }B",
            r"\ =\ 18",
            r"\ \text{(vs. 54 on the GPU).}",
            font_size=26, color=HI_COLOR,
        )
        # Multi-line Trainium note (Text with an explicit newline so it
        # fits comfortably in the right half of the frame).
        trainium_note = Text(
            "AWS Trainium's NeuronCore:\n"
            "a 128×128 PE grid — 16,384\n"
            "MACs/cycle, each input read ONCE.",
            font_size=22, color=HI_COLOR, line_spacing=0.9,
        )

        # Stack them on the right side, well clear of the PE grid.
        right_stack = VGroup(reads, trainium_note).arrange(
            DOWN, buff=0.55, aligned_edge=LEFT,
        )
        right_stack.move_to(np.array([3.0, -0.3, 0]))

        self.play(Write(reads))
        self.wait(1.0)
        self.play(FadeIn(trainium_note))
        self.wait(2.4)

        # ---------- 6i. Fade out everything scene-local ----------
        self.play(
            FadeOut(pe_group),
            FadeOut(result_title),
            FadeOut(reads),
            FadeOut(trainium_note),
            FadeOut(caption),
        )

    # ==================================================================
    # 7. Trade-offs table
    # ==================================================================
    def _compare(self) -> None:
        caption = Text(
            "Step 7:  Two architectures, same workload — different trade-offs.",
            font_size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))

        headers = VGroup(
            Text("Aspect", font_size=24, color=WHITE),
            Text("GPU (SIMT)", font_size=24, color=BLUE),
            Text("Systolic Array", font_size=24, color=GREEN),
        )

        rows = [
            ("Parallelism",
             "Thousands of threads",
             "Fixed N×N PE grid"),
            ("Memory reads (our 3×3)",
             "54",
             "18"),
            ("Data movement",
             "Threads ↔ register file",
             "PE → neighbour PE"),
            ("Energy / MAC",
             "Higher",
             "Much lower"),
            ("Flexibility",
             "Any kernel, any shape",
             "Fixed shapes; great for matmul/conv"),
            ("Programming model",
             "CUDA / Triton",
             "Compiler-scheduled (e.g. NKI)"),
        ]

        # Build a table of Text mobjects
        all_rows = VGroup()
        all_rows.add(headers)
        for r in rows:
            row = VGroup(
                Text(r[0], font_size=20, color=WHITE),
                Text(r[1], font_size=20, color=BLUE),
                Text(r[2], font_size=20, color=GREEN),
            )
            all_rows.add(row)

        # Arrange as a grid
        col_widths = [3.4, 3.8, 4.8]
        for row in all_rows:
            for i, cell in enumerate(row):
                cell.set_x(-5 + sum(col_widths[:i]) + col_widths[i] / 2 - 1.5)
        all_rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        # Re-apply x positions after arrange (arrange resets x)
        for row in all_rows:
            cells = list(row)
            for i, cell in enumerate(cells):
                x = -5.5 + sum(col_widths[:i]) + col_widths[i] / 2
                cell.set_x(x)
        all_rows.move_to(UP * 0.1)

        # Separator line under header
        sep = Line(
            start=all_rows.get_left() + DOWN * 0.05,
            end=all_rows.get_right() + DOWN * 0.05,
            color=GREY_B, stroke_width=1,
        )
        sep.next_to(all_rows[0], DOWN, buff=0.08)
        sep.set_x(all_rows.get_center()[0])

        self.play(FadeIn(all_rows[0]))
        self.play(Create(sep))
        for row in all_rows[1:]:
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.35)

        takeaway = Text(
            "GPUs are general-purpose workhorses.  Systolic arrays are matmul specialists.",
            font_size=24, color=HI_COLOR,
        ).next_to(caption, UP, buff=0.15)
        self.play(FadeIn(takeaway))
        self.wait(2.6)

        self.play(
            FadeOut(all_rows), FadeOut(sep),
            FadeOut(takeaway), FadeOut(caption),
        )

    # ==================================================================
    # 8. Recap
    # ==================================================================
    def _summary(self) -> None:
        self.play(FadeOut(self.roadmap))

        title = Text("Recap", font_size=48, color=YELLOW).to_edge(UP, buff=0.8)
        lines = VGroup(
            Text("1.  By hand: 27 multiply-adds, one at a time", font_size=26),
            Text("2.  CPU: 1 ALU → 27 sequential cycles", font_size=26),
            Text("3.  GPU (SIMT): 9 threads in lockstep → 3 cycles,"
                 "   but 54 memory reads", font_size=26),
            Text("4.  Systolic array: 3 cycles of useful work"
                 "   with only 18 reads — each value used 3×", font_size=26),
        )
        punch = MathTex(
            r"\text{GPU: brute-force parallelism.}\quad"
            r"\text{Systolic: choreographed dataflow.}",
            font_size=32,
            color=HI_COLOR,
        )
        lines.add(punch)
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(
            title, DOWN, buff=0.55
        )

        tagline = Text(
            "Why accelerators matter:  less memory traffic = less energy = more FLOPs/watt.",
            font_size=24,
            color=HI_COLOR,
        ).to_edge(DOWN, buff=0.8)

        self.play(Write(title))
        for line in lines:
            self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.5)
        self.play(Write(tagline))
        self.wait(3.0)
        self.play(FadeOut(title), FadeOut(lines), FadeOut(tagline))
