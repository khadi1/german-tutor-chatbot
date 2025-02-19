from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch

app = Flask(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")



model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2",  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

@app.route("/ask", methods=["POST"])
def ask_german_assistant():
    data = request.json
    question = data.get("question", "")

    # Tokenize input and move to GPU
    inputs = tokenizer(question, return_tensors="pt").to(model.device) 

    # Generate an answer
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            max_length=50,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
             pad_token_id=tokenizer.eos_token_id,
             repetition_penalty=1.2
        )

    # Decode response
    answer = tokenizer.decode(output[0])

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=False)