import os
from pathlib import Path


def get_project_path():
    """
    return project root directory
    :return:
    """
    return Path(__file__).parent.parent
# print("s")
# print(get_project_path())
