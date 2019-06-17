from django.shortcuts import render
from django.http import HttpResponse

def home(request):
	return render(request=request, template_name='MLDashboard/index.html', context={'title': 'Flight System'})
