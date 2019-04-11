import sys
from pathlib import Path
from shutil import copytree, ignore_patterns


# This script initializes new pytorch project with the template files.
# Run `python3 new_project.py ../MyNewProject` then new project named 
# MyNewProject will be made
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: python3 new_project.py MyNewProject'
project_name = Path(sys.argv[1])

ignore = [".git", "data", "saved", "new_project.py", "LICENSE", ".flake8", "README.md", "__pycache__"]
copytree('.', project_name, ignore=ignore_patterns(*ignore))
