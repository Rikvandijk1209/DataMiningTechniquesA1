"""Script used to make sure the notebooks have acces to the code in src/"""

import os
import sys


def get_root_path(target_dir: str) -> str:
    """Get the root path of the given target_dir."""
    # Get the current file path
    current_path = os.path.abspath(__file__)

    # Split the path into components
    path_components = current_path.split(os.sep)

    # Find the index of the target directory
    if target_dir in path_components:
        target_index = path_components.index(target_dir)

        # Join the components up to the target directory
        root_path = os.sep.join(path_components[: target_index + 1])
        return root_path
    else:
        raise ValueError(f"Directory '{target_dir}' not found in the path.")


PROJECT_ROOT = get_root_path("sparo")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    print(f"Added the path ({PROJECT_ROOT}) to sys.path")
else:
    print(f"Path ({PROJECT_ROOT}) already exists in sys.path")
