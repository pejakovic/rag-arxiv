import os
from pathlib import Path
from typing import Union, Optional

class PathManager:
    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize PathManager with a root directory.
        If not provided, uses the parent directory of the 'src' folder.
        """
        if root_dir is None:
            # Get the directory containing the src folder
            root_dir = str(Path(__file__).parent.parent)
        self.root_dir = Path(root_dir).resolve()

    def get_abs_path(self, path: Union[str, Path], create_dirs: bool = False) -> Path:
        """
        Convert a relative path to absolute path based on root directory.
        If the path is already absolute, return it as is.
        
        Args:
            path: The path to convert
            create_dirs: If True, creates the directory if it doesn't exist
            
        Returns:
            Path: Absolute path
        """
        path = Path(path)
        
        # If path is absolute, return it as is
        if path.is_absolute():
            abs_path = path
        else:
            # Convert relative path to absolute using root_dir
            abs_path = (self.root_dir / path).resolve()
        
        # Create directories if requested
        if create_dirs and not abs_path.parent.exists():
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
        return abs_path

    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """
        Ensure directory exists and return its absolute path.
        Creates the directory if it doesn't exist.
        """
        abs_path = self.get_abs_path(path)
        abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path 