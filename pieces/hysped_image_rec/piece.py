from domino.base_piece import BasePiece
from domino.types import PieceOutput, Inputs, Outputs

class HySpedImageRecPiece(BasePiece):
    def piece_function(self, inputs: Inputs) -> Outputs:
        image_path = inputs.get("image_path", "/workspace/sample.jpg")
        self.logger.info(f"Running HySpedImageRecPiece on {image_path}")
        return PieceOutput({"result": f"Processed {image_path}"})
