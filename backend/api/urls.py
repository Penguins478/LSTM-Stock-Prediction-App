from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('predict/', views.generate_prediction_graph, name='predict'),
    # Add other URL patterns for your project here if needed
]
