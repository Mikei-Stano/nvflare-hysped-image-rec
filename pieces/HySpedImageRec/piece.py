from .models import HySpedInput, HySpedOutput

def main(input_model: HySpedInput) -> HySpedOutput:
    # TODO: integrate your real inference here
    preds = {"status": "ok", "note": "placeholder – integrate NVFlare/HySped inference"}
    return HySpedOutput(predictions=preds)
