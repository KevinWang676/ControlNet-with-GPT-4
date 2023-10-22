# 🌁🪄🌃 - ControlNet with GPT-4
## 🌟 Born to Create: Controllable Text-to-Image Generation with GPT-4

### Quick start: [Colab Notebook](https://colab.research.google.com/github/KevinWang676/ControlNet-with-GPT-4/blob/main/ControlNet_with_GPT_4.ipynb) ⚡
**Hugging Face demo**: [ControlNet with GPT-4](https://huggingface.co/spaces/kevinwang676/ControlNet-with-GPT-4) 🤗 (Need a GPU)

GPT-4 can generate code from a prompt, which can be rendered as an image, in a way that is true to the instructions to a higher degree of accuracy. This project is inspired by the [paper](https://arxiv.org/abs/2303.12712) - *Sparks of Artificial General Intelligence: Early experiments with GPT-4*.

P.S. You may need to have a GPT-4 API key since GPT-3.5 would not work properly. If you would like to run `app.py` locally, you need to specify `torch==1.13.1` in [`requirements.txt`](https://github.com/KevinWang676/ControlNet-with-GPT-4/blob/main/requirements.txt).

## Comparison between Stable Diffusion 2.1 and ControlNet with GPT-4

*Prompt: a soccer ball to the right of a television and to the left of a cup, and they are all on a desk*

**Stable Diffusion 2.1:**

![image](https://github.com/KevinWang676/ControlNet-with-GPT-4/assets/126712357/46fbfcb2-6820-4a98-945f-be3484277471)

**ControlNet with GPT-4:**

![image](https://github.com/KevinWang676/ControlNet-with-GPT-4/assets/126712357/24f89c70-6e17-4c4f-b383-f882e4855936)

## Gradio Interface

**Colab Interface:**

![image](https://github.com/KevinWang676/ControlNet-with-GPT-4/assets/126712357/0ff99fb5-3bb0-46fe-af6a-348e262f0791)

**Hugging Face Interface:**

![916d1cb085564851de75dc67a18f8ef](https://github.com/KevinWang676/ControlNet-with-GPT-4/assets/126712357/ad3d5854-bd11-4273-aee7-481b03ba1a9e)
