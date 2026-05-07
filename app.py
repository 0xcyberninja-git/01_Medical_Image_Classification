import gradio as gr
from transformers import pipeline

# Load the deepfake detection model
pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

def detect_deepfake(image):
    """
    Classifies an image as Realism or Deepfake using the pre-trained model.
    """
    if image is None:
        return None

    # Run the model on the uploaded image
    results = pipe(image)

    # Format the results for Gradio's Label component
    # The pipeline returns a list of dicts: [{'label': 'Realism', 'score': 0.9}, ...]
    output = {result['label']: float(result['score']) for result in results}
    return output

# Create the Gradio interface
with gr.Blocks(title="Deepfake Detection Tool") as demo:
    gr.Markdown("# 🕵️‍♂️ Deepfake Detection Tool")
    gr.Markdown("Upload an image to detect whether it is real or a deepfake. This tool uses a pre-trained Vision Transformer model to classify images.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column():
            label_output = gr.Label(num_top_classes=2, label="Prediction Confidence")

    # Add some example images if possible or just keep it simple

    submit_btn.click(
        fn=detect_deepfake,
        inputs=image_input,
        outputs=label_output
    )

if __name__ == "__main__":
    demo.launch()
