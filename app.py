__all__ = ['is_cat','learn','classify_image','categories','image','label','examples','intf']

import gradio as gr
from fastai.vision.all import *


def is_cat(x): return x[0].isupper()

learn = load_learner('imageclass.pkl')

categories = ('Cat','Mice')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image(type="pil", image_mode="RGB", height=192, width=192)
label = gr.Label()

examples = ['cat.jpg','mice.jpg']

intf = gr.Interface(fn=classify_image, inputs=image,outputs=label, examples=examples)
intf.launch(inline=False)