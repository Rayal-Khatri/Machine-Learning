from django.urls import path
from Ml_App.views import predict
from.import views

urlpatterns = [
    path('',views.index,name='index'),
    path('predict/', predict, name='predict'),
]
