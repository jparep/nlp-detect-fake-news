# core/urls.py

from django.contrib import admin
from django.urls import path
from core.views import home  # Corrected import
from api.views import PredictView

urlpatterns = [
    path('', home, name='home'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('admin/', admin.site.urls),
]
