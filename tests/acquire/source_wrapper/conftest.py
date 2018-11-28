import pytest
from sksurgeryimage.acquire import source_wrapper


@pytest.fixture(scope="function")
def video_source():
    return source_wrapper.VideoSourceWrapper()
