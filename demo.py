import gradio as gr
import numpy as np
import time

"""
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

def fake_diffusion(steps):
    rng = np.random.default_rng()
    for i in range(steps):
        time.sleep(1)
        image = rng.random(size=(600, 600, 3))
        yield image
    image = np.ones((1000,1000,3), np.uint8)
    image[:] = [255, 124, 0]
    yield image

def joint_interface(name, intensity, steps):
    greeting_output = greet(name, intensity)
    diffusion_output = fake_diffusion(steps)

    return greeting_output, diffusion_output

demo = gr.Interface(
    fn=joint_interface,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Slider(1, 10, 3, step=1, label="Intensity"),
        gr.Slider(1, 10, 3, step=1, label="Steps")
    ],
    outputs=[
        gr.Textbox(label="Greeting Output"),
        "image"
    ]
)

demo.launch()




def yes_man(message, history):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"

gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="Yes Man",
    description="Ask Yes Man any question",
    theme="soft",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()

"""

def echo(message, history, system_prompt, tokens):
    response = f"System prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i+1]

with gr.Blocks() as demo:
    system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
    slider = gr.Slider(10, 100, render=False)

    gr.ChatInterface(
        echo, additional_inputs=[system_prompt, slider]
    )

demo.launch()

