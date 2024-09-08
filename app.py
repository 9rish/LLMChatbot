from flask import Flask, request, jsonify
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load model and tokenizer
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

import logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    message = data.get("message", "")

    # Generate a response
    inputs = tokenizer.encode(message, return_tensors="tf")
    outputs = model.generate(
    inputs, 
    max_length=50, 
    num_return_sequences=1, 
    no_repeat_ngram_size=2  # Adjust this parameter to reduce repetition
)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
