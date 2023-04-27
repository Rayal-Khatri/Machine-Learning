from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from Ml_App.employee_ml import predict_late_tomorrow, get_employee_name
from datetime import datetime, timedelta

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        prediction = predict_late_tomorrow(user_id)
        employee_name = get_employee_name(user_id)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A")
        return JsonResponse({'prediction': prediction, 'name': employee_name, 'tomorrow': tomorrow})
    else:
        return render(request, 'predict.html')

def index(request):
    return render(request,'index.html')


