from flask import Flask
from textblob import TextBlob

# Create Flask app
app = Flask(__name__)

# a simple home page
@app.route('/')
def hello():
    return 'Python is good in VS Tool Box...!'

# Create sentiment api
@app.route('/<message>')
def inbox(message):
    sentiment = 'positive'
    if(TextBlob(message).sentiment.polarity < 0):
        sentiment = 'negitive'
    return app.make_response(sentiment)