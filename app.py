import gradio as gr

prompt_tips = '''
The \<A\> and \<B\> are the first and second entities appeared in the prompt respectively. The \<R\> is the relationship between them.
'''
def create_view():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                exemplar_img = gr.Image(
                    label='Original Stable Diffusion Output',
                    type='pil',
                    interactive=False
                )
                gr.Markdown(prompt_tips)
                prompt = gr.Textbox(
                    label='Prompt',
                    max_lines=1,
                    placeholder='A bottle contains a car')
                A = gr.Textbox(
                    label='<A>',
                    max_lines=1,
                    placeholder='bottle')
                R = gr.Dropdown(
                    label='<R>',
                    choices=["contain","carry","below","place"],
                    value=0
                    )
                B = gr.Textbox(
                    label='<B>',
                    max_lines=1,
                    placeholder='car')



                run_button = gr.Button('Generate')

            with gr.Column():
                result = gr.Image(label='Result', interactive=False)
                adjustment_scale = gr.Slider(label='Scale of relationship adjustment',
                                               minimum=0,
                                               maximum=0.8,
                                               step=0.1,
                                               value=0.6)
                guidance_scale = gr.Slider(label='Classifier-Free Guidance Scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=7.5)
                ddim_steps = gr.Slider(label='Number of DDIM Sampling Steps',
                                               minimum=10,
                                               maximum=100,
                                               step=1,
                                               value=50)


#         prompt.submit(
#             fn=func,
#             # inputs=[
#             #     model_id,
#             #     prompt,
#             #     num_samples,
#             #     guidance_scale,
#             #     ddim_steps
#             # ],
#             inputs=[
#                 prompt,
#                 num_samples,
#                 guidance_scale,
#                 ddim_steps
#             ],
#             outputs=result,
#             queue=False
#         )

#         run_button.click(
#             fn=func,
#             inputs=[
#                 exemplar_dataset,
#                 prompt,
#                 num_samples,
#                 guidance_scale,
#                 ddim_steps
#             ],
#             outputs=result,
#             queue=False
#         )
    return demo

TITLE = '# RRNET'
DESCRIPTION = '''
This is a gradio demo for **Relation Rectification in Diffusion Model**
'''
# DETAILDESCRIPTION='''
# RRNET
# '''
# DETAILDESCRIPTION='''
# We propose a new task, **Relation Inversion**: Given a few exemplar images, where a relation co-exists in every image, we aim to find a relation prompt **\<R>** to capture this interaction, and apply the relation to new entities to synthesize new scenes.
# Here we give several pre-trained relation prompts for you to play with. You can choose a set of exemplar images from the examples, and use **\<R>** in your prompt for relation-specific text-to-image generation.
# '''
with gr.Blocks(css='style.css') as demo:
    # if not torch.cuda.is_available():
    #     show_warning(CUDA_NOT_AVAILABLE_WARNING)

    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    # gr.Markdown(DETAILDESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('Relation-Specific Text-to-Image Generation'):
            create_view()

demo.queue().launch(share=True, server_port=6006)
