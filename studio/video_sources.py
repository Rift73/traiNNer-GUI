"""Stream each frame-role member selected by paired video windows once per scene."""

from collections.abc import Iterator

from trainner_data.frame_sequences import FrameScene


def selected_frame_positions(
    scene: FrameScene, width: int
) -> Iterator[tuple[str, str, int, int]]:
    """Yield role, folder, frame offset and first using window (one-based in scene).

    The ordered union skips sparse gaps. Each target belongs only to its center
    window. No context list is allocated or traversed again for adjacent windows.
    """
    assert scene.target_root is not None
    covered = 0
    for first, start in enumerate(scene.starts, 1):
        for pos in range(max(start, covered), start + width):
            yield "lq", scene.root, pos, first
        covered = start + width
        yield "gt", scene.target_root, start + width // 2, first
