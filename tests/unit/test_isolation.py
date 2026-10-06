"""The suite must not read or write the developer's real files.

Engines build a CacheManager rooted at ~/.cache/whisperbox even in test mode
(and prune it on startup), and entry points load the project .env.
"""

import os

import src.config as config_module


def test_home_is_a_temp_dir(tmp_path):
    assert os.path.expanduser("~").startswith(str(tmp_path))


def test_project_env_file_is_not_the_real_one(tmp_path):
    assert not config_module.DEFAULT_ENV_FILE.exists()
