from flask import Flask, render_template, request, jsonify
import pyttsx3
import speech_recognition as sr
import torch
import random
import json
import google.generativeai as genai
from utils.model_utils import load_model, load_label_encoder, load_vectorizer
from utils.data_utils import load_knowledge_base, find_product_details
from threading import Thread
import subprocess
import time
from PIL import Image
import io
import boto3

app = Flask(__name__)

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Tokenization function
def tokenize(text):
    return text.split()

# Load model and preprocessors
model_path = 'models/intent_classifier.pth'
label_encoder_path = 'models/label_encoder.pkl'
vectorizer_path = 'models/vectorizer.pkl'
intents_path = 'data/intents.json'
knowledge_base_path = 'data/knowledge_base.json'
user_generated_intents_path = 'data/user_generated_intents.json'

label_encoder = load_label_encoder(label_encoder_path)
vectorizer = load_vectorizer(vectorizer_path, tokenize)
with open(intents_path, 'r') as f:
    intents = json.load(f)['intents']
knowledge_base = load_knowledge_base(knowledge_base_path)

input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)

model = load_model(model_path, input_dim, output_dim)
model.eval()

# Gemini API Configuration
API_KEY = "As if, add you API key man"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# AWS Rekognition Configuration
AWS_ACCESS_KEY_ID = 'Noop Not a chance Add your API key'
AWS_SECRET_ACCESS_KEY = 'its a secret so add yours'
AWS_REGION = 'us-east-2'

rekognition_client = boto3.client('rekognition',
                                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                  region_name=AWS_REGION)

# Function to get a response from Gemini API
def get_gemini_response(question):
    chat = gemini_model.start_chat(history=[])
    response = chat.send_message(question)
    return response.text, response

def get_response(user_input):
    X = vectorizer.transform([user_input]).toarray()
    inputs = torch.tensor(X, dtype=torch.float32)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    tag = label_encoder.inverse_transform(predicted.numpy())[0]
    return tag

def log_interaction(user_input, response, tag="user_generated"):
    new_data = {
        "patterns": [user_input],
        "responses": [response],
        "tag": tag
    }
    try:
        with open(user_generated_intents_path, 'r') as file:
            user_generated_intents = json.load(file)
    except FileNotFoundError:
        user_generated_intents = {"intents": []}

    # Check if the tag already exists
    for intent in user_generated_intents['intents']:
        if intent['tag'] == tag:
            intent['patterns'].append(user_input)
            intent['responses'].append(response)
            break
    else:
        user_generated_intents['intents'].append(new_data)

    with open(user_generated_intents_path, 'w') as file:
        json.dump(user_generated_intents, file, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get('text')
    engine.say(text)
    engine.runAndWait()
    return jsonify({'status': 'success'})

@app.route('/listen', methods=['POST'])
def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return jsonify({'text': text})
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand the audio'})
    except sr.RequestError as e:
        return jsonify({'error': f'Could not request results; {e}'})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    tag = get_response(user_input)

    print(f"User input: {user_input}")
    print(f"Predicted tag: {tag}")

    response = "Sorry, I couldn't understand your query."
    if tag == 'product_query':
        product_name = user_input.split("about")[-1].strip()  # Extract product name from user input
        print(f"Product name extracted: {product_name}")
        response = find_product_details(product_name, knowledge_base)
        if response == "Sorry, I couldn't find any details for that product.":
            response, gemini_response = get_gemini_response(user_input)
            response += "\n\n(This information is gathered from Gemini API)"
            log_interaction(user_input, response, tag="product_query")
    elif tag == 'noans':
        response, gemini_response = get_gemini_response(user_input)
        if "tech" in gemini_response.text.lower() or "pc parts" in gemini_response.text.lower():
            response += "\n\n(This information is gathered from Gemini API)"
            gemini_tag = gemini_response.text.split("\n\n")[0].replace("**Tag:** ", "")
            log_interaction(user_input, response, tag=gemini_tag)
        else:
            response = "I'm sorry, I'm only allowed to answer tech-related and PC parts questions. :)"
            log_interaction(user_input, response, tag="non_tech_questions")
    else:
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                break

    print(f"Response: {response}")
    return jsonify({'response': response})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'response': "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'response': "No selected file"}), 400

    if file:
        img = Image.open(file.stream)
        img = img.convert('RGB')  # Ensure image is in RGB format

        # Determine the format based on the file extension or content type
        img_format = 'JPEG' if file.mimetype == 'image/jpeg' else 'PNG'

        # Save the image to a byte array to send to Amazon Rekognition
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        img_byte_arr.seek(0)

        response = rekognition_client.detect_labels(
            Image={'Bytes': img_byte_arr.getvalue()},
            MaxLabels=10
        )

        labels = response['Labels']
        print(f"Detected labels: {labels}")
        relevant_labels = {'Computer', 'Laptop', 'PC', 'Hardware', 'Keyboard', 'Monitor', 'Mouse', 'Graphics Card', 'CPU', 'Motherboard'}
        detected_labels = {label['Name'] for label in labels}

        matching_labels = relevant_labels.intersection(detected_labels)
        if matching_labels:
            response_message = f"The image is of a PC part or computer-related. Detected labels: {', '.join(matching_labels)}"
        else:
            response_message = "I'm sorry, I think the image is not a PC part or computer-related. Try uploading a different image."

        return jsonify({'response': response_message})

def classify_image(image):
    # This function is now integrated with Amazon Rekognition, no need for the random choice
    pass

# Function to run the Jupyter notebook
def run_notebook():
    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 'notebooks/train_model.ipynb'])

# Function to run the notebook periodically
def run_notebook_periodically(interval=600):
    while True:
        run_notebook()
        time.sleep(interval)

# Start the notebook execution thread
notebook_thread = Thread(target=run_notebook_periodically, args=(600,))
notebook_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
