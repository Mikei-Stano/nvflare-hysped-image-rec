from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HySpedInput:
    image_path: str  # absolute or workspace-relative path

@dataclass
class HySpedOutput:
    predictions: Dict[str, Any]
