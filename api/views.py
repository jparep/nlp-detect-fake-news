# api/views.py

from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import joblib
import os

# Load the model and vectorizer at the start
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/model/model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '../ml/model/vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    def post(self, request):
        text = request.POST.get('text', '')
        prediction = None
        if text:
            try:
                news_vector = vectorizer.transform([text])
                print(f"Vectorized input: {news_vector}")
                prediction = model.predict(news_vector)[0]
                print(f"Prediction: {prediction}")
                if prediction == 1:
                    result = 'Fake News'
                else:
                    result = 'Real News'
            except Exception as e:
                print(f"Error during prediction: {e}")
                result = "Error predicting the news type"
            return render(request, 'index.html', {'prediction': result})
        return render(request, 'index.html', {'prediction': 'No text provided'})
