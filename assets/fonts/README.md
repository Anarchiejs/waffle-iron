# Fonts

`lucide.ttf` (Lucide 0.544.0) is included. The interface merges six private-use
icons from it (library, cpu, gauge, git-branch, columns-2, activity). The ISC
license and the MIT notice for the Feather icons it incorporates are in
`LICENSE-lucide.txt` in this folder and in `NOTICE` at the repository root.
https://github.com/lucide-icons/lucide

The CMake build copies this directory next to `WaffleIron` for every preset
that builds the interface, and `cmake --install` installs it beside the binary.

Inter is optional. Place `Inter-Regular.ttf` and `Inter-SemiBold.ttf` here,
from https://github.com/rsms/inter (SIL Open Font License 1.1). Do not commit
those files.

Space Grotesk and JetBrains Mono are optional too, under the same license and
the same rule (do not commit them): `SpaceGrotesk-SemiBold.ttf` (or
`SpaceGrotesk-Medium.ttf`) for page titles, from
https://github.com/floriankarsten/space-grotesk, and `JetBrainsMono-Regular.ttf`
for numbers and paths, from https://github.com/JetBrains/JetBrainsMono.
Without them, titles use the semibold text font and values use a system
monospace font.

Without Inter, Waffle Iron uses a system font (Segoe UI on Windows, DejaVu or
Liberation on Linux, Arial on macOS) and otherwise ImGui's built-in font.
Without `lucide.ttf` the sidebar draws simple strokes instead of private-use
glyphs.
