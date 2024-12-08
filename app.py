import gradio as gr
from huggingface_hub import InferenceClient
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import transformers

options=['Empathetic','Helpful','Understanding']

def saveData(text, options, output):
  import csv
  with open(#your file here!, 'a') as f:
    w=csv.writer(f)
    w.writerow([text, options, output])


def starchat(text, theme):
  modelname = "HuggingFaceH4/starchat2-15b-v0.1"
  client=InferenceClient(model=modelname, token=# your token)
  thiscontent=f'give only one {options} response for this statement: {text}?'
  messages=[{'role': 'user', 'content':thiscontent},]
  output=client.chat_completion(messages, max_tokens=100)
  response=output.choices[0].message.content
  saveData(text, options, response)
  return response

def zephyr(text, theme):
  modelname = "HuggingFaceH4/zephyr-7b-beta"
  client=InferenceClient(model=modelname, token=#your token)
  thiscontent=f'give only one {options} response for this statement: {text}?'
  messages=[{'role': 'user', 'content':thiscontent},]
  output=client.chat_completion(messages, max_tokens=100)
  response=output.choices[0].message.content
  saveData(text, options, response)
  return response

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    with gr.Row():
      with gr.Column():
        text = gr.Textbox(label="What\'s on your mind",scale=1)
        theme=gr.Dropdown(options, label="How do you want to be helped?")
      with gr.Column():
        zephyr_btn = gr.Button("Zephyr, say something!",scale=1)
        output = gr.Textbox(label="Zephyr assistant says:",scale=1)
        zephyr_btn.click(fn=zephyr, inputs=[text, theme], outputs=output)
      with gr.Column():
        starchat_btn = gr.Button("Starchat, say something!",scale=1)
        output = gr.Textbox(label="Starchat assistant says:",scale=1)
        starchat_btn.click(fn=starchat, inputs=[text, theme], outputs=output)
demo.launch(debug=True)
