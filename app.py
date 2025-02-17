from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",  load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

@app.route("/ask", methods=["POST"])
def ask_german_assistant():
    data = request.json
    question = data.get("question", "")
    
   
    inputs = tokenizer(question, return_tensors="pt")
    
    # Move tensors to the same device as the model
    input_ids = inputs["input_ids"].to(model.device)
    
    # Generate an answer
    with torch.no_grad():
        output = model.generate(input_ids)
    
    # Decode the generated response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)