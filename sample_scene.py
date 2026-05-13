"""Sample Manim scene used to verify the local installation.

Render from the project root (with the `manim-env` conda env activated) via:

    manim -pql manim_scripts/sample_scene.py HelloManim

Flags:
    -p  : preview the rendered file after it finishes
    -q  : quality (l=low, m=medium, h=high, k=4k)
    -l  : (legacy alias for low quality — kept here for convenience)

Output files land in `./media/` by default.
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    PI,
    YELLOW,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    Scene,
    Square,
    Text,
    Transform,
    Write,
)


class HelloManim(Scene):
    """A small demo scene that exercises shapes, text, and transforms."""

    def construct(self) -> None:
        title = Text("Hello, Manim!", font_size=56, color=YELLOW)
        subtitle = Text(
            "Local environment is ready 🎬",
            font_size=28,
            color=BLUE,
        ).next_to(title, DOWN, buff=0.5)

        # Intro text
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=DOWN))
        self.wait(0.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Shape transform demo
        square = Square(side_length=2.0, color=BLUE).shift(3 * LEFT)
        circle = Circle(radius=1.0, color=GREEN).set_fill(GREEN, opacity=0.5)

        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(square.animate.rotate(PI / 2))
        self.wait(0.5)
        self.play(FadeOut(square))
