from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare the dataset
def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

def load_data():
    ham = read_category('ham', '../enron1/ham')
    spam = read_category('spam', '../enron1/spam')
    df = pd.DataFrame.from_records(ham + spam)
    df['content'] = df['content'].apply(preprocessor)
    return df

# Text preprocessing and stop words
def preprocessor(e):
    return re.sub('[^A-Za-z]', ' ', e).lower()

custom_stop_words = [
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 
    'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself', 
    'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 'this', 'that', 'these', 'those', 'who', 
    'whom', 'whose', 'which', 'that', 'anyone', 'everyone', 'someone', 'no one', 'anybody', 'everybody', 
    'somebody', 'nobody', 'anything', 'everything', 'something', 'nothing', 'all', 'each', 'few', 'many', 
    'none', 'some', 'one', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 
    'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 
    'concerning', 'considering', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 
    'like', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over', 'past', 'regarding', 'round', 
    'since', 'through', 'throughout', 'till', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 
    'within', 'without', 'the', 'a', 'an', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'although', 'because', 
    'since', 'unless', 'while', 'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'can', 'could', 'shall', 'should', 'will', 'would', 'may', 'might', 
    'must', 'ought', 'go', 'come', 'get', 'make', 'take', 'give', 'say', 'know', 'see', 'think', 'want', 'use', 
    'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', "subject", "not", "no", "more", "here", 
    "any", "if", "only", "please"
]

df = load_data()
vectorizer = CountVectorizer(preprocessor=preprocessor, stop_words=custom_stop_words)
X = vectorizer.fit_transform(df['content'])
y = df['category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced')
model.fit(X_train_scaled, y_train)

@app.route('/')
def prediction():
    return render_template('prediction.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text')
    if not input_text:
        return 'No input text provided', 400

    
    processed_text = preprocessor(input_text)
    input_vector = vectorizer.transform([processed_text])
    input_scaled = scaler.transform(input_vector)
    
    
    prediction = model.predict(input_scaled)[0]
    category = 'Spam' if prediction == 1 else 'Ham'
    return f'The input text is classified as: {category}'

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        content = file.read().decode('utf-8')  
        processed_text = preprocessor(content) 
        input_vector = vectorizer.transform([processed_text])
        input_scaled = scaler.transform(input_vector)
        
        prediction = model.predict(input_scaled)[0]
        category = 'Spam' if prediction == 1 else 'Ham'
        
        return f'The uploaded file is classified as: {category}', 200

if __name__ == '__main__':
    app.run(debug=True)
