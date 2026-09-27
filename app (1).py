import gradio as gr
from rembg import remove
from PIL import Image

def remove_bg(image: Image.Image):
    if image is None:
        return None
    return remove(image)

demo = gr.Interface(
    fn=remove_bg,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Image(type="pil", label="Background removed", image_mode="RGBA"),
    title="AI Background Remover",
    description="Upload a photo and get a clean, transparent-background PNG using a real trained AI segmentation model (U2-Net), running on the server.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
