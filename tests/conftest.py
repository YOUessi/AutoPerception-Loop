from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_image(tmp_path: Path):
    def factory(name: str = "frame.jpg", size: tuple[int, int] = (100, 50)) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size=size, color=(40, 80, 120)).save(path)
        return path

    return factory
