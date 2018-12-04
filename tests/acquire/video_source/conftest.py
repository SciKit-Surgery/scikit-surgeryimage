import pytest
from sksurgeryimage.acquire import video_source as vs


@pytest.fixture(scope="function")
def video_source():
    return vs.VideoSourceWrapper()
