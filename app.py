import os
import gradio as gr
from rembg import remove, new_session
from PIL import Image

# u2netp is a much smaller/lighter model (~5MB) than the default u2net
# (~176MB), so it fits inside Render's free-tier 512MB memory limit.
session = new_session("u2netp")

def remove_bg(image: Image.Image):
    if image is None:
        return None
    return remove(image, session=session)

demo = gr.Interface(
    fn=remove_bg,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Image(type="pil", label="Background removed", image_mode="RGBA"),
    title="AI Background Remover",
    description="Upload a photo and get a clean, transparent-background PNG using a real trained AI segmentation model (U2-Net, lightweight version), running on the server.",
    allow_flagging="never",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
