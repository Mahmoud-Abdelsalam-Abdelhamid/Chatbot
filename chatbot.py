import random
import json
import numpy as np
import joblib
import torch

from NN_model import NeuralNetwork
import nlp_utils
from transformers import pipeline

import os
import cohere

data = torch.load("F:\Programming\GDG\Chatbot\self task\data_0.94_0.94.pth")
pipeline = joblib.load("F:\Programming\GDG\Chatbot\self task\pipeline.pkl")
scaler = joblib.load("F:\Programming\GDG\Chatbot\self task\scaler.pkl")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_response(prompt):
    co = cohere.ClientV2("k4nwDC2goIUbxM6QKGQYjw4Yo8APJyihG648PXsQ")
    response = co.chat(
        model="command-r-plus", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content[0].text


with open('F:\Programming\GDG\Chatbot\self task\intents.json', 'r') as json_data:
    intents = json.load(json_data)


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Intent Bot"
print("Let's chat! (type 'quit' to exit)")

DEBUG = False  # Set to True to enable debug prints

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    # Transform the new sentence
    processed_sentence = np.array(pipeline.transform([sentence]))
    processed_sentence = scaler.transform(processed_sentence)
    processed_sentence = torch.tensor(processed_sentence, dtype=torch.float32).to(device)

    # Ensure correct shape (batch size, features)
    if len(processed_sentence.shape) == 1:
        processed_sentence = processed_sentence.unsqueeze(0)

    output = model(processed_sentence)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")
                print(tag)

                if DEBUG:
                    print(f"DEBUG: User input -> {sentence}")
                    print(f"DEBUG: Processed input shape -> {processed_sentence.shape}")
                    print(f"DEBUG: Model output -> {output}")
                    print(f"DEBUG: Predicted tag -> {tag}, Probability -> {prob.item()}")

    else:
        print(f"AI Bot: {generate_response(sentence)}")
