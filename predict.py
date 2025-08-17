from cog import BasePredictor, Input, Path
from video_inference import process_video, setup_onnx_model

class Predictor(BasePredictor):
    def setup(self) -> None:
        # Initialize ONNX Runtime session with fine-tuned model
        model_path = 'yolov11n-UAV-finetune.onnx'
        input_name, session = setup_onnx_model(model_path)

        self.model_path = model_path
        self.session = session
        self.input_name = input_name

    def predict(
        self,
        video: Path = Input(description="Drone Swarm Input video file")
    ) -> Path:
        return process_video(video, self.input_name, self.session, self.model_path)
