"""MLDashboard URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='Flight Home'),
    url(r'^barGraph/', TemplateView.as_view(template_name="MLDashboard/barGraph.html"), name='barGraph'),
    url(r'^barGraph2/', TemplateView.as_view(template_name="MLDashboard/barGraph2.html"), name='barGraph2'),
    url(r'^barGraph3/', TemplateView.as_view(template_name="MLDashboard/barGraph3.html"), name='barGraph3'),
]
