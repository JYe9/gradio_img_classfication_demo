---
title: Img Classfication
emoji: 👀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
license: mit 
---

The live demo is [here](https://huggingface.co/spaces/jye9/img_classfication)

## Introduction

In this tutorial, we will explore how to create a simple and interactive Object Recognition application using the Gradio library, we will use the Microsoft BEIT model from Hugging Face. Gradio is a Python library that allows you to rapidly create UIs for machine learning models, making it easy to demonstrate and interact with your models.

## Prerequisites for run in local machine

Before we begin, make sure you have the following installed and set up:

1. **Hugging Face Account:**
    - Ensure you have a Hugging Face account. If not, you can create one [here](https://huggingface.co/join).
2. **Gradio and Transformers:**
    - Install the necessary libraries by running the following commands in your terminal:
        
        ```bash
        pip install gradio transformers
        
        ```
        

## Step 1: Import Libraries

Start by importing the required libraries in your Python script.

```python
import gradio as gr
from transformers import pipeline

```

## Step 2: Load the BEIT Model

Load the Microsoft BEIT model using the Hugging Face Transformers library.

```python
# Load the BEIT model for image classification
pipe = pipeline(task="image-classification",
                model="microsoft/beit-base-patch16-384")

```

## Step 3: Create Gradio Interface

Use Gradio to create a simple UI for object recognition.

```python
# Create Gradio interface from the pipeline
gr.Interface.from_pipeline(pipe,
                           title="Image Classification",
                           description="Object Recognition using Microsoft BEIT",
                           examples=['egyptian_cat.jpg', 'German_shepherd.jpg']
                           ).launch()

```

## Step 4: Customize the Interface (Optional)

You can customize the interface by adding more features or modifying the appearance. For example, you can include an article section with the author's information.

```python
# Add article information to the interface
gr.Interface.from_pipeline(pipe,
                           title="Image Classification",
                           description="Object Recognition using Microsoft BEIT",
                           examples=['egyptian_cat.jpg', 'German_shepherd.jpg'],
                           ).launch()

```

## Step 5: Launch the Application

Finally, launch the Gradio application and interact with it.

```bash
# Run the script in your terminal
python app.py

```

Visit the provided URL in your browser to use the interactive Object Recognition application.
