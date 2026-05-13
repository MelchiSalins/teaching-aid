"""Tiny MathTex smoke test — renders a frame using LaTeX.

Run:
    manim -s -ql manim_scripts/latex_smoke.py LatexSmoke
(-s = save the last frame as PNG; no video needed.)
"""

from manim import BLUE, YELLOW, DOWN, MathTex, Scene, Text, Write


class LatexSmoke(Scene):
    def construct(self) -> None:
        title = Text("LaTeX is wired up!", font_size=40, color=YELLOW)
        eq = MathTex(
            r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left("
            r"\frac{QK^{\top}}{\sqrt{d}}\right) V",
            font_size=48,
            color=BLUE,
        ).next_to(title, DOWN, buff=0.6)
        self.play(Write(title))
        self.play(Write(eq))
        self.wait(0.5)
