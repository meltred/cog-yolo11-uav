import gradio as gr
from video_inference import setup_onnx_model, process_video

def run_model(video):
    onnxFile = 'yolov11n-UAV-finetune.onnx'
    input_name, session = setup_onnx_model(onnxFile)
    output = process_video(session, video, input_name, onnxFile)
    return output

inference = gr.Interface(
    fn=run_model, 
    inputs=gr.Video(), 
    outputs=gr.Video()
)

inference.launch()