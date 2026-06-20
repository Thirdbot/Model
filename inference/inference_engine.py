import gradio as gr


def run_inference(prompt, system_prompt, max_tokens, temperature):
    """Swap this stub with your tokenizer/model generate call."""
    prompt = prompt.strip()
    system_prompt = system_prompt.strip()

    if not prompt:
        return "Write a prompt first."

    return (
        "Model output will appear here.\n\n"
        f"System: {system_prompt or 'None'}\n"
        f"Prompt: {prompt}\n"
        f"Settings: max_tokens={max_tokens}, temperature={temperature}"
    )


theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size="md",
)

with gr.Blocks(theme=theme, title="Inference Engine") as app:
    gr.Markdown("Inference Engine")

    with gr.Row():
        with gr.Column(scale=2):
            system_prompt = gr.Textbox(
                label="System prompt",
                placeholder="You are a precise, helpful assistant.",
                lines=3,
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Ask the model something...",
                lines=8,
            )

            with gr.Row():
                max_tokens = gr.Slider(
                    32,
                    2048,
                    value=512,
                    step=32,
                    label="Max tokens",
                )
                temperature = gr.Slider(
                    0.0,
                    1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )

            submit = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Output", lines=18)

    gr.Examples(
        examples=[
            ["Summarize why LoRA is useful for fine-tuning.", "", 256, 0.4],
            ["Write a short Python function that batches a list.", "", 256, 0.2],
        ],
        inputs=[prompt, system_prompt, max_tokens, temperature],
    )

    submit.click(
        fn=run_inference,
        inputs=[prompt, system_prompt, max_tokens, temperature],
        outputs=output,
        api_name="predict",
    )


if __name__ == "__main__":
    app.launch()
