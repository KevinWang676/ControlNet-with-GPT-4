from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
import re
import openai
from cairosvg import svg2png

# Constants
low_threshold = 100
high_threshold = 200

# Models
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# This command loads the individual model components on GPU on-demand. So, we don't
# need to explicitly call pipe.to("cuda").
pipe.enable_model_cpu_offload()

pipe.enable_xformers_memory_efficient_attention()

# Generator seed,
generator = torch.manual_seed(0)

def get_canny_filter(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def generate_images(image, prompt):
    canny_image = get_canny_filter(image)
    output = pipe(
        prompt,
        canny_image,
        generator=generator,
        num_images_per_prompt=3,
        num_inference_steps=20,
    )
    all_outputs = []
    all_outputs.append(canny_image)
    for image in output.images:
        all_outputs.append(image)
    return all_outputs

# GPT-4 control

def gpt_control(apikey, prompt):

    openai.api_key = apikey

    # gpt
    messages = [{"role": "system", "content": "You are an SVG expert with years of experience and multiple contributions to the SVG project. Based on the prompt and the description, please generate the corresponding SVG code."},
     {"role": "user", "content": f"""Provide only the shell command without any explanations.
The current objective is below. Reply with the SVG code only:
OBJECTIVE: {prompt}
YOUR SVG CODE:
"""}]

    completion = openai.ChatCompletion.create(
      model = "gpt-4",
      messages = messages
    )

    chat_response = completion.choices[0].message.content

    code = re.findall('<.*>', chat_response)
    code_new = '\n'.join(code)

    svg_code = f"""
    {code_new}
    """
    svg2png(bytestring=svg_code,write_to='output.png')

    return 'output.png'

with gr.Blocks(theme="JohnSmith9982/small_and_pretty") as app:
  gr.HTML("<center>"
          "<h1>🌁🪄🌃 - ControlNet with GPT-4</h1>"
          "</center>")

  gr.Markdown("## <center>🌟 Born to Create: Controllable Text-to-Image Generation with GPT-4</center>")

  with gr.Row():
    with gr.Column():
      inp1 = gr.Textbox(label="OpenAI API Key", type="password")
      inp2 = gr.Textbox(label="Position Prompt (as simple as possible)")
      inp3 = gr.Textbox(label="Image Prompt (make it shine)")
      btn1 = gr.Button("GPT-4 Control", variant="primary")
      btn2 = gr.Button("Generate", variant="primary")
    with gr.Column():
      out1 = gr.Image(label="Output Image", type="pil")
      out2 = gr.Gallery(label="Generated Images").style(grid=[2], height="auto")
  btn1.click(gpt_control, [inp1, inp2], [out1])
  btn2.click(generate_images, [out1, inp3], [out2])

app.launch(show_error=True, share=True)
