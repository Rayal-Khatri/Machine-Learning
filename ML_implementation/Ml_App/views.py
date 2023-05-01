from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from datetime import datetime, timedelta
from Ml_App.employee_ml import get_prediction

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        prediction, employee_name = get_prediction(user_id)
        prediction = round(prediction, 2)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A")
        return JsonResponse({'prediction': prediction, 'name': employee_name, 'tomorrow': tomorrow})
    else:
        return render(request, 'predict.html')

def index(request):
    return render(request,'index.html')


