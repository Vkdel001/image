from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from PIL import Image
import os

app = Flask(__name__)

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token='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'
).to(device)

def generate_image(prompt, filename="generated_image.png"):
    with autocast(device):
        image = pipe(prompt).images[0]
    image_path = os.path.join(os.getcwd(), filename)
    image.save(image_path)
    print(f"Image saved at: {image_path}")  # Debug statement
    return image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    image_path = generate_image(prompt)
    if os.path.exists(image_path):
        return send_file(image_path, as_attachment=True)
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
