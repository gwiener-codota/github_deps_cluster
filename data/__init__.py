import os


_my_dir = os.path.dirname(__file__)


def get(*path):
    return os.path.join(_my_dir, *path)
