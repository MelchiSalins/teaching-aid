"""LLM data-flow teaching animation.

Walks an IT-savvy audience through what happens inside a (decoder-only)
Large Language Model when it processes a word:

    token  ->  embedding  ->  + positional  ->  Q/K/V  ->  attention  ->  KV cache

Render from the project root (with the ``manim-env`` conda env active and
BasicTeX available on PATH):

    # quick preview while iterating
    manim -pql manim_scripts/llm_dataflow.py LLMDataFlow
    # final high-quality render
    manim -pqh manim_scripts/llm_dataflow.py LLMDataFlow
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_E,
    DOWN,
    GREEN_E,
    GREY,
    GREY_B,
    LEFT,
    ORANGE,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Create,
    FadeIn,
    FadeOut,
    Indicate,
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
# Colour vocabulary (consistent across every sub-scene)
# ---------------------------------------------------------------------------
Q_COLOR = BLUE
K_COLOR = "#2ecc71"       # a clear green
V_COLOR = ORANGE
HI_COLOR = YELLOW
CACHE_COLOR = GREY_B

SENTENCE = ["The", "cat", "sat", "down"]
TOKEN_IDS = [101, 464, 782, 319]
SEQ_LEN = len(SENTENCE)
D_MODEL = 4

RNG = np.random.default_rng(7)

# Pre-generate all numbers once so they stay consistent between sub-scenes.
TOKEN_EMB = np.round(RNG.uniform(-1, 1, size=(SEQ_LEN, D_MODEL)), 2)
POS_EMB = np.round(
    np.stack(
        [
            np.sin(np.arange(D_MODEL) / D_MODEL + i * 0.7)
            for i in range(SEQ_LEN)
        ]
    ),
    2,
)
X = np.round(TOKEN_EMB + POS_EMB, 2)
W_Q = np.round(RNG.uniform(-0.6, 0.6, size=(D_MODEL, D_MODEL)), 2)
W_K = np.round(RNG.uniform(-0.6, 0.6, size=(D_MODEL, D_MODEL)), 2)
W_V = np.round(RNG.uniform(-0.6, 0.6, size=(D_MODEL, D_MODEL)), 2)
Q = np.round(X @ W_Q, 2)
K = np.round(X @ W_K, 2)
V = np.round(X @ W_V, 2)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def vector_mob(values, color=WHITE, scale=0.75):
    """Return a proper column-vector Matrix mobject (LaTeX-rendered)."""
    entries = [[f"{v:.2f}"] for v in values]
    m = Matrix(entries, h_buff=0.7, v_buff=0.55, bracket_h_buff=0.15)
    m.set_color(color)
    m.scale(scale)
    return m


def matrix_mob(values, color=WHITE, scale=0.75):
    """Return a proper (rectangular) Matrix mobject (LaTeX-rendered)."""
    entries = [[str(c) for c in row] for row in values]
    m = Matrix(entries, h_buff=0.85, v_buff=0.55, bracket_h_buff=0.15)
    m.set_color(color)
    m.scale(scale)
    return m


def token_pill(text: str, color=BLUE_E, width=1.4, height=0.6):
    box = Rectangle(width=width, height=height, color=color, fill_opacity=0.25)
    label = Text(text, font_size=24, color=WHITE)
    label.move_to(box.get_center())
    return VGroup(box, label)


# ---------------------------------------------------------------------------
# The main scene
# ---------------------------------------------------------------------------
class LLMDataFlow(Scene):
    """Teaching animation: from a word to the KV cache."""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def construct(self) -> None:
        self._title_card()
        self.roadmap = self._build_roadmap()
        self.add(self.roadmap)

        self._highlight_step(0)
        self._tokenize()

        self._highlight_step(1)
        self._embed()

        self._highlight_step(2)
        self._positional()

        self._highlight_step(3)
        self._qkv()

        self._highlight_step(4)
        self._attention()

        self._highlight_step(5)
        self._kv_cache()

        self._summary()

    # ------------------------------------------------------------------
    # 1. Title
    # ------------------------------------------------------------------
    def _title_card(self) -> None:
        title = Text("From Word to KV Cache", font_size=56, color=YELLOW)
        sub = Text(
            "How an LLM actually processes a token",
            font_size=28,
            color=BLUE,
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(title))
        self.play(FadeIn(sub, shift=DOWN))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub))

    # ------------------------------------------------------------------
    # Roadmap / breadcrumb at top
    # ------------------------------------------------------------------
    def _build_roadmap(self) -> VGroup:
        steps_text = [
            "Tokenize",
            "Embedding",
            "+ Positional",
            "Q / K / V",
            "Attention",
            "KV Cache",
        ]
        pills = VGroup()
        for s in steps_text:
            p = token_pill(s, color=GREY, width=1.9, height=0.55)
            pills.add(p)
        pills.arrange(RIGHT, buff=0.2)
        pills.to_edge(UP, buff=0.3)
        arrows = VGroup()
        for a, b in zip(pills[:-1], pills[1:]):
            arr = Arrow(
                a.get_right(),
                b.get_left(),
                buff=0.02,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(arr)
        self.steps = pills
        return VGroup(pills, arrows)

    def _highlight_step(self, idx: int) -> None:
        anims = []
        for i, pill in enumerate(self.steps):
            box, label = pill
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
        self.play(*anims, run_time=0.6)

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    def _tokenize(self) -> None:
        caption = Text(
            "Step 1:  The text is split into tokens, each with an integer ID.",
            font_size=26,
        ).to_edge(DOWN, buff=0.4)

        words = VGroup(*[Text(w, font_size=40) for w in SENTENCE])
        words.arrange(RIGHT, buff=0.35)
        words.move_to(UP * 1.3)

        self.play(Write(words), FadeIn(caption))
        self.wait(0.6)

        pills = VGroup()
        ids_mobs = VGroup()
        for i, word in enumerate(SENTENCE):
            p = token_pill(word, color=BLUE_E, width=1.4, height=0.7)
            p.move_to(words[i].get_center() + DOWN * 2)
            id_txt = Text(f"id = {TOKEN_IDS[i]}", font_size=22, color=GREY_B)
            id_txt.next_to(p, DOWN, buff=0.15)
            pills.add(p)
            ids_mobs.add(id_txt)

        self.play(
            *[TransformFromCopy(words[i], pills[i]) for i in range(SEQ_LEN)],
            run_time=1.2,
        )
        self.play(FadeIn(ids_mobs, shift=DOWN * 0.2))
        self.wait(1.4)

        self.token_pills = pills
        self.token_ids = ids_mobs

        self.play(FadeOut(words), FadeOut(caption))

    # ------------------------------------------------------------------
    # 3. Token -> Embedding vector
    # ------------------------------------------------------------------
    def _embed(self) -> None:
        caption = Text(
            "Step 2:  Each ID looks up a row in the embedding matrix.",
            font_size=26,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption))

        pills_group = VGroup(self.token_pills, self.token_ids)
        self.play(
            pills_group.animate.scale(0.8).to_edge(LEFT, buff=0.4).shift(UP * 0.3)
        )

        focus_pill = self.token_pills[1]
        focus_id = self.token_ids[1]
        focus_ring = SurroundingRectangle(
            VGroup(focus_pill, focus_id), color=HI_COLOR, buff=0.08
        )
        self.play(Create(focus_ring))

        rows = [
            [r"\vdots", r"\vdots", r"\vdots", r"\vdots"],
            [f"{v:.2f}" for v in TOKEN_EMB[1]],
            [r"\vdots", r"\vdots", r"\vdots", r"\vdots"],
        ]
        emb_matrix = matrix_mob(rows, color=WHITE, scale=0.8)
        emb_matrix.shift(RIGHT * 0.3)

        emb_label = Text("Embedding matrix", font_size=22, color=GREY_B)
        emb_label.next_to(emb_matrix, UP, buff=0.2)
        id_annot = Text("row 464", font_size=22, color=HI_COLOR)
        id_annot.next_to(emb_matrix, LEFT, buff=0.3).shift(DOWN * 0.05)

        self.play(FadeIn(emb_matrix), FadeIn(emb_label))

        # Highlight the "cat" row
        row_cells = VGroup(*emb_matrix.get_rows()[1])
        highlight = SurroundingRectangle(row_cells, color=HI_COLOR, buff=0.1)
        self.play(Create(highlight), FadeIn(id_annot))
        self.wait(0.6)

        cat_vec = vector_mob(TOKEN_EMB[1], color=HI_COLOR, scale=0.85)
        cat_vec.to_edge(RIGHT, buff=1.2)
        vec_label = MathTex("e_{\\mathrm{cat}}", color=HI_COLOR).next_to(
            cat_vec, UP, buff=0.2
        )

        self.play(TransformFromCopy(row_cells, cat_vec), Write(vec_label))
        self.wait(1.2)

        self.play(
            FadeOut(emb_matrix),
            FadeOut(emb_label),
            FadeOut(highlight),
            FadeOut(id_annot),
            FadeOut(focus_ring),
            FadeOut(caption),
        )
        self.cat_emb_vec = cat_vec
        self.cat_emb_label = vec_label

    # ------------------------------------------------------------------
    # 4. Add positional embedding
    # ------------------------------------------------------------------
    def _positional(self) -> None:
        caption = Text(
            "Step 3:  Add a positional vector so the model knows word ORDER.",
            font_size=26,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption))

        target = LEFT * 3.4
        self.play(
            self.cat_emb_vec.animate.move_to(target),
            self.cat_emb_label.animate.next_to(target, UP, buff=0.3),
        )

        plus = MathTex("+", font_size=48).next_to(
            self.cat_emb_vec, RIGHT, buff=0.45
        )
        pos_vec = vector_mob(POS_EMB[1], color=BLUE, scale=0.85)
        pos_vec.next_to(plus, RIGHT, buff=0.45)
        pos_label = MathTex("p_{1}", color=BLUE).next_to(pos_vec, UP, buff=0.3)

        self.play(Write(plus), FadeIn(pos_vec), Write(pos_label))
        self.wait(0.5)

        eq = MathTex("=", font_size=48).next_to(pos_vec, RIGHT, buff=0.45)
        x_vec = vector_mob(X[1], color=WHITE, scale=0.85)
        x_vec.next_to(eq, RIGHT, buff=0.45)
        x_label = MathTex("x_{1}", color=WHITE).next_to(x_vec, UP, buff=0.3)

        self.play(Write(eq))
        self.play(FadeIn(x_vec), Write(x_label))
        self.wait(1.2)

        # Show all 4 position-aware vectors in a row
        all_x = VGroup()
        for i in range(SEQ_LEN):
            xi = vector_mob(X[i], color=WHITE, scale=0.55)
            lbl = MathTex(f"x_{{{i}}}", color=WHITE).scale(0.7).next_to(
                xi, DOWN, buff=0.15
            )
            all_x.add(VGroup(xi, lbl))
        all_x.arrange(RIGHT, buff=0.5).shift(DOWN * 0.2)

        self.play(
            FadeOut(self.cat_emb_vec),
            FadeOut(self.cat_emb_label),
            FadeOut(plus),
            FadeOut(pos_vec),
            FadeOut(pos_label),
            FadeOut(eq),
            FadeOut(x_vec),
            FadeOut(x_label),
            FadeOut(self.token_pills),
            FadeOut(self.token_ids),
        )
        self.play(FadeIn(all_x))
        note = Text(
            "Every token now has a position-aware vector.",
            font_size=22,
            color=GREY_B,
        ).next_to(all_x, UP, buff=0.5)
        self.play(FadeIn(note))
        self.wait(1.4)

        self.all_x = all_x
        self.play(FadeOut(note), FadeOut(caption))

    # ------------------------------------------------------------------
    # 5. Q / K / V projections
    # ------------------------------------------------------------------
    def _qkv(self) -> None:
        caption = Text(
            "Step 4:  Multiply each x by three weight matrices to get Q, K, V.",
            font_size=26,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption))

        self.play(
            self.all_x.animate.scale(0.75).to_edge(LEFT, buff=0.4).shift(UP * 1.5)
        )

        wq = self._weight_tile("W_Q", Q_COLOR)
        wk = self._weight_tile("W_K", K_COLOR)
        wv = self._weight_tile("W_V", V_COLOR)
        weights = VGroup(wq, wk, wv).arrange(DOWN, buff=0.2)
        weights.shift(RIGHT * 0.2 + DOWN * 0.2)
        self.play(FadeIn(weights, shift=RIGHT * 0.3))
        self.wait(0.3)

        x1 = self.all_x[1]
        x1_ring = SurroundingRectangle(x1, color=HI_COLOR, buff=0.08)
        self.play(Create(x1_ring))

        q1 = vector_mob(Q[1], color=Q_COLOR, scale=0.7)
        k1 = vector_mob(K[1], color=K_COLOR, scale=0.7)
        v1 = vector_mob(V[1], color=V_COLOR, scale=0.7)
        q1_lbl = MathTex("q_{1}", color=Q_COLOR).next_to(q1, UP, buff=0.15)
        k1_lbl = MathTex("k_{1}", color=K_COLOR).next_to(k1, UP, buff=0.15)
        v1_lbl = MathTex("v_{1}", color=V_COLOR).next_to(v1, UP, buff=0.15)

        triple = VGroup(
            VGroup(q1, q1_lbl),
            VGroup(k1, k1_lbl),
            VGroup(v1, v1_lbl),
        ).arrange(RIGHT, buff=0.55)
        triple.to_edge(RIGHT, buff=0.6).shift(DOWN * 0.3)

        self.play(
            TransformFromCopy(x1, q1),
            TransformFromCopy(x1, k1),
            TransformFromCopy(x1, v1),
            FadeIn(q1_lbl),
            FadeIn(k1_lbl),
            FadeIn(v1_lbl),
            Indicate(wq[0], color=Q_COLOR),
            Indicate(wk[0], color=K_COLOR),
            Indicate(wv[0], color=V_COLOR),
            run_time=1.8,
        )
        self.wait(0.6)

        formula = MathTex(
            r"q_i = x_i W_Q \quad k_i = x_i W_K \quad v_i = x_i W_V",
            font_size=34,
        ).next_to(triple, DOWN, buff=0.6)
        self.play(Write(formula))
        self.wait(1.6)

        # Build the full Q, K, V "stacks" (small vectors in a row)
        def stack(matrix, color, label_tex):
            vecs = VGroup(
                *[vector_mob(matrix[i], color=color, scale=0.45) for i in range(SEQ_LEN)]
            )
            vecs.arrange(RIGHT, buff=0.3)
            lbl = MathTex(label_tex, color=color).scale(1.0).next_to(
                vecs, LEFT, buff=0.3
            )
            return VGroup(vecs, lbl)

        Q_stack = stack(Q, Q_COLOR, "Q")
        K_stack = stack(K, K_COLOR, "K")
        V_stack = stack(V, V_COLOR, "V")
        stacks = VGroup(Q_stack, K_stack, V_stack).arrange(DOWN, buff=0.3)
        stacks.move_to(DOWN * 0.3)

        self.play(
            FadeOut(x1_ring),
            FadeOut(weights),
            FadeOut(formula),
            FadeOut(triple),
            FadeOut(self.all_x),
        )
        self.play(FadeIn(stacks))
        note = Text(
            "Done in parallel for every token -> full Q, K, V matrices.",
            font_size=22,
            color=GREY_B,
        ).next_to(stacks, UP, buff=0.4)
        self.play(FadeIn(note))
        self.wait(1.6)

        self.play(FadeOut(stacks), FadeOut(note), FadeOut(caption))

    def _weight_tile(self, tex_name, color):
        box = Rectangle(width=1.6, height=0.9, color=color, fill_opacity=0.25)
        label = MathTex(tex_name, color=color).scale(1.1).move_to(box.get_center())
        return VGroup(box, label)

    # ------------------------------------------------------------------
    # 6. Self-attention — the core calculation
    # ------------------------------------------------------------------
    def _attention(self) -> None:
        caption = Text(
            "Step 5:  Each query asks every key 'how relevant are you?'",
            font_size=26,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption))

        formula = MathTex(
            r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left("
            r"\frac{QK^{\top}}{\sqrt{d}}\right) V",
            font_size=36,
        ).to_edge(UP, buff=1.4)
        self.play(Write(formula))
        self.wait(0.6)

        # Queries on the left (column)
        q_vecs = VGroup()
        for i in range(SEQ_LEN):
            v = vector_mob(Q[i], color=Q_COLOR, scale=0.45)
            lbl = MathTex(f"q_{{{i}}}", color=Q_COLOR).scale(0.7).next_to(
                v, LEFT, buff=0.15
            )
            q_vecs.add(VGroup(v, lbl))
        q_vecs.arrange(DOWN, buff=0.18).to_edge(LEFT, buff=0.8).shift(DOWN * 0.3)

        # Keys on the top
        k_vecs = VGroup()
        for i in range(SEQ_LEN):
            v = vector_mob(K[i], color=K_COLOR, scale=0.45)
            lbl = MathTex(f"k_{{{i}}}", color=K_COLOR).scale(0.7).next_to(
                v, UP, buff=0.1
            )
            k_vecs.add(VGroup(v, lbl))
        k_vecs.arrange(RIGHT, buff=0.3).to_edge(UP, buff=2.2).shift(RIGHT * 0.6)

        # Values on the bottom
        v_vecs = VGroup()
        for i in range(SEQ_LEN):
            v = vector_mob(V[i], color=V_COLOR, scale=0.45)
            lbl = MathTex(f"v_{{{i}}}", color=V_COLOR).scale(0.7).next_to(
                v, DOWN, buff=0.1
            )
            v_vecs.add(VGroup(v, lbl))
        v_vecs.arrange(RIGHT, buff=0.3).to_edge(DOWN, buff=1.3).shift(RIGHT * 0.6)

        self.play(FadeIn(q_vecs), FadeIn(k_vecs), FadeIn(v_vecs))
        self.wait(0.3)

        # Compute causal-masked softmax attention weights
        scores_raw = Q @ K.T / np.sqrt(D_MODEL)
        mask = np.triu(np.ones_like(scores_raw) * -1e9, k=1)
        scores_masked = scores_raw + mask
        attn = np.exp(scores_masked - scores_masked.max(axis=1, keepdims=True))
        attn = attn / attn.sum(axis=1, keepdims=True)

        for q_idx in [1, 2, 3]:
            self._single_attention_step(q_idx, q_vecs, k_vecs, v_vecs, attn[q_idx])

        mask_note = Text(
            "Causal mask: a token can only attend to itself and earlier tokens.",
            font_size=22,
            color=GREY_B,
        ).next_to(caption, UP, buff=0.2)
        self.play(FadeIn(mask_note))
        self.wait(1.4)

        self.play(
            FadeOut(formula),
            FadeOut(q_vecs),
            FadeOut(k_vecs),
            FadeOut(v_vecs),
            FadeOut(mask_note),
            FadeOut(caption),
        )

    def _single_attention_step(self, q_idx, q_vecs, k_vecs, v_vecs, weights):
        q_obj = q_vecs[q_idx]
        q_ring = SurroundingRectangle(q_obj, color=HI_COLOR, buff=0.08)
        self.play(Create(q_ring), run_time=0.5)

        arrows = VGroup()
        for i in range(SEQ_LEN):
            color = HI_COLOR if weights[i] > 1e-6 else GREY
            arr = Arrow(
                q_obj.get_top(),
                k_vecs[i].get_bottom(),
                buff=0.1,
                stroke_width=2,
                color=color,
                max_tip_length_to_length_ratio=0.08,
            )
            arrows.add(arr)
        self.play(*[Create(a) for a in arrows], run_time=0.8)

        # Softmax bar chart
        bars = VGroup()
        bar_labels = VGroup()
        for i, w in enumerate(weights):
            h = max(0.03, float(w) * 1.0)
            active = w > 1e-6
            b = Rectangle(
                width=0.45,
                height=h,
                color=HI_COLOR if active else GREY,
                fill_opacity=0.85 if active else 0.15,
            )
            bars.add(b)
            lbl = MathTex(f"{w:.2f}", color=WHITE).scale(0.55)
            bar_labels.add(lbl)
        bars.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        bars.next_to(v_vecs, UP, buff=0.15)
        for b, lbl in zip(bars, bar_labels):
            lbl.next_to(b, UP, buff=0.05)

        softmax_caption = MathTex(
            rf"\mathrm{{softmax}}\!\left(q_{{{q_idx}}} K^{{\top}} / \sqrt{{d}}\right)",
            font_size=30,
            color=HI_COLOR,
        ).next_to(bars, UP, buff=0.2)

        self.play(FadeIn(bars), FadeIn(bar_labels), FadeIn(softmax_caption))
        self.wait(0.5)

        # Weighted sum of V -> output
        out_numeric = np.sum(
            [weights[i] * V[i] for i in range(SEQ_LEN)], axis=0
        )
        out_vec = vector_mob(out_numeric, color=HI_COLOR, scale=0.5)
        out_lbl = MathTex(f"o_{{{q_idx}}}", color=HI_COLOR).next_to(
            out_vec, UP, buff=0.1
        )
        out_group = VGroup(out_vec, out_lbl).to_edge(RIGHT, buff=0.4).shift(
            DOWN * 0.2
        )

        weighted_arrows = VGroup()
        for i in range(SEQ_LEN):
            if weights[i] < 1e-6:
                continue
            arr = Arrow(
                v_vecs[i].get_top(),
                out_vec.get_bottom(),
                buff=0.1,
                stroke_width=2 + 4 * float(weights[i]),
                color=V_COLOR,
                max_tip_length_to_length_ratio=0.08,
            )
            weighted_arrows.add(arr)
        self.play(
            *[Create(a) for a in weighted_arrows],
            FadeIn(out_group),
            run_time=0.9,
        )
        self.wait(0.7)

        self.play(
            FadeOut(arrows),
            FadeOut(bars),
            FadeOut(bar_labels),
            FadeOut(softmax_caption),
            FadeOut(weighted_arrows),
            FadeOut(out_group),
            FadeOut(q_ring),
            run_time=0.6,
        )

    # ------------------------------------------------------------------
    # 7. KV cache population during autoregressive generation
    # ------------------------------------------------------------------
    def _kv_cache(self) -> None:
        caption = Text(
            "Step 6:  During generation, K and V for past tokens are CACHED.",
            font_size=26,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption))

        subtitle = Text(
            "Each new token only computes its OWN q, k, v — old ones are reused.",
            font_size=22,
            color=GREY_B,
        ).next_to(caption, UP, buff=0.15)
        self.play(FadeIn(subtitle))

        cache_slots = 4
        k_boxes = VGroup(
            *[
                Rectangle(width=1.0, height=0.45, color=K_COLOR, fill_opacity=0.05)
                for _ in range(cache_slots)
            ]
        )
        v_boxes = VGroup(
            *[
                Rectangle(width=1.0, height=0.45, color=V_COLOR, fill_opacity=0.05)
                for _ in range(cache_slots)
            ]
        )
        k_boxes.arrange(DOWN, buff=0.08)
        v_boxes.arrange(DOWN, buff=0.08)

        k_title = Text("K-cache", font_size=22, color=K_COLOR).next_to(
            k_boxes, UP, buff=0.15
        )
        v_title = Text("V-cache", font_size=22, color=V_COLOR).next_to(
            v_boxes, UP, buff=0.15
        )
        k_group = VGroup(k_title, k_boxes)
        v_group = VGroup(v_title, v_boxes)
        cache_panel = VGroup(k_group, v_group).arrange(RIGHT, buff=0.35)
        cache_panel.to_edge(RIGHT, buff=0.6).shift(UP * 0.2)

        panel_frame = SurroundingRectangle(cache_panel, color=CACHE_COLOR, buff=0.25)
        panel_lbl = Text("KV Cache", font_size=24, color=WHITE).next_to(
            panel_frame, UP, buff=0.1
        )
        self.play(FadeIn(panel_frame), FadeIn(panel_lbl), FadeIn(cache_panel))
        self.wait(0.4)

        gen_title = Text("Generation step:", font_size=24, color=WHITE)
        gen_title.to_edge(LEFT, buff=0.6).shift(UP * 2.2)
        self.play(FadeIn(gen_title))

        step_tokens = ["The", "cat", "sat"]
        step_indices = [0, 1, 2]

        for step_num, (tok, idx) in enumerate(zip(step_tokens, step_indices)):
            step_label = Text(
                f"t = {step_num + 1}:  new token = \"{tok}\"",
                font_size=26,
                color=HI_COLOR,
            ).next_to(gen_title, DOWN, buff=0.35)

            pill = token_pill(tok, color=BLUE_E, width=1.3, height=0.55)
            pill.next_to(step_label, DOWN, buff=0.4)

            self.play(FadeIn(step_label), FadeIn(pill))
            self.wait(0.25)

            qi = vector_mob(Q[idx], color=Q_COLOR, scale=0.5)
            ki = vector_mob(K[idx], color=K_COLOR, scale=0.5)
            vi = vector_mob(V[idx], color=V_COLOR, scale=0.5)
            qi_lbl = MathTex(f"q_{{{idx}}}", color=Q_COLOR).next_to(qi, UP, buff=0.1)
            ki_lbl = MathTex(f"k_{{{idx}}}", color=K_COLOR).next_to(ki, UP, buff=0.1)
            vi_lbl = MathTex(f"v_{{{idx}}}", color=V_COLOR).next_to(vi, UP, buff=0.1)

            triple = VGroup(
                VGroup(qi, qi_lbl),
                VGroup(ki, ki_lbl),
                VGroup(vi, vi_lbl),
            ).arrange(RIGHT, buff=0.4)
            triple.next_to(pill, DOWN, buff=0.5)

            self.play(
                TransformFromCopy(pill, qi),
                TransformFromCopy(pill, ki),
                TransformFromCopy(pill, vi),
                FadeIn(qi_lbl),
                FadeIn(ki_lbl),
                FadeIn(vi_lbl),
                run_time=1.0,
            )
            self.wait(0.3)

            target_k = k_boxes[idx]
            target_v = v_boxes[idx]

            stored_k = ki.copy().scale(0.55).move_to(target_k.get_center())
            stored_v = vi.copy().scale(0.55).move_to(target_v.get_center())

            fill_k = target_k.copy().set_fill(K_COLOR, opacity=0.35).set_stroke(K_COLOR)
            fill_v = target_v.copy().set_fill(V_COLOR, opacity=0.35).set_stroke(V_COLOR)

            self.play(
                Transform(ki, stored_k),
                Transform(vi, stored_v),
                Transform(target_k, fill_k),
                Transform(target_v, fill_v),
                FadeOut(ki_lbl),
                FadeOut(vi_lbl),
                run_time=1.1,
            )
            q_used = Text("used → discarded", font_size=20, color=GREY_B).next_to(
                qi, DOWN, buff=0.15
            )
            self.play(FadeIn(q_used))
            self.wait(0.3)
            self.play(FadeOut(qi), FadeOut(qi_lbl), FadeOut(q_used))

            if step_num >= 1:
                reuse_txt = Text(
                    f"Reused from cache: slots 0..{step_num - 1}",
                    font_size=20,
                    color=CACHE_COLOR,
                ).next_to(pill, RIGHT, buff=0.6)
                self.play(FadeIn(reuse_txt))
                cached_k = VGroup(*[k_boxes[j] for j in range(step_num)])
                cached_v = VGroup(*[v_boxes[j] for j in range(step_num)])
                self.play(
                    Indicate(cached_k, color=CACHE_COLOR),
                    Indicate(cached_v, color=CACHE_COLOR),
                    run_time=0.9,
                )
                self.wait(0.3)
                self.play(FadeOut(reuse_txt))

            self.play(
                FadeOut(step_label),
                FadeOut(pill),
                FadeOut(ki),
                FadeOut(vi),
                run_time=0.5,
            )

        punch1 = MathTex(
            r"\text{Without cache: } \mathcal{O}(n^{2})"
            r"\;\; \text{— recompute } K, V \text{ for ALL past tokens.}",
            font_size=28,
            color=GREY_B,
        )
        punch2 = MathTex(
            r"\text{With KV cache: } \mathcal{O}(n)"
            r"\;\; \text{— compute one new } k, v\text{; rest is a lookup.}",
            font_size=28,
            color=HI_COLOR,
        )
        punch = VGroup(punch1, punch2).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        punch.next_to(gen_title, DOWN, buff=0.3).align_to(gen_title, LEFT)

        self.play(FadeIn(punch))
        self.wait(2.4)

        self.play(
            FadeOut(punch),
            FadeOut(gen_title),
            FadeOut(cache_panel),
            FadeOut(panel_frame),
            FadeOut(panel_lbl),
            FadeOut(subtitle),
            FadeOut(caption),
        )

    # ------------------------------------------------------------------
    # 8. Summary card
    # ------------------------------------------------------------------
    def _summary(self) -> None:
        self.play(FadeOut(self.roadmap))

        title = Text("Recap", font_size=48, color=YELLOW).to_edge(UP, buff=0.8)
        lines = VGroup(
            Text("1.  Text -> tokens -> integer IDs", font_size=28),
            Text("2.  IDs -> embedding vectors (learned lookup)", font_size=28),
            Text("3.  + positional encoding -> x_i knows WHERE it is", font_size=28),
            Text("4.  Project x_i with W_Q, W_K, W_V -> q_i, k_i, v_i", font_size=28),
        )
        # Append a LaTeX-rendered line for the Attention formula
        attn_line = MathTex(
            r"\text{5.}\;\; \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right) V"
            r"\;\; \text{mixes information}",
            font_size=34,
        )
        lines.add(attn_line)
        lines.add(
            Text(
                "6.  At inference: keep k_i, v_i in the KV cache forever",
                font_size=28,
                color=HI_COLOR,
            )
        )
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(
            title, DOWN, buff=0.6
        )

        tagline = Text(
            "KV cache = why LLM inference can stream tokens fast.",
            font_size=26,
            color=HI_COLOR,
        ).to_edge(DOWN, buff=0.8)

        self.play(Write(title))
        for line in lines:
            self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.4)
        self.play(Write(tagline))
        self.wait(2.5)
        self.play(FadeOut(title), FadeOut(lines), FadeOut(tagline))
