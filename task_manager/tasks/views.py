from django.http import HttpResponse

def home(request):
    return HttpResponse("Tasks app working")



