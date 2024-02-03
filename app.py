import gradio as gr
from transformers import pipeline

pipe = pipeline(task="image-classification", 
                # model that can do category classification
                # https://huggingface.co/microsoft/beit-base-patch16-384
                model = "microsoft/beit-base-patch16-384")
gr.Interface.from_pipeline(pipe, 
                           title="Image Classification",
                           description="Object Recognition using Microsoft BEIT",
                           examples = ['egyptian_cat.jpg', 'German_shepherd.jpg',],
                           ).launch()
