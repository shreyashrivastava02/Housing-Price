'''
Created on Jul 17, 2019

@author: shreya0008
'''
from django.contrib import admin
from django.urls import path
from django.urls.conf import include
from Housing import views
from django.views.generic.base import RedirectView
from rest_framework import routers


router = routers.DefaultRouter()
router.register(r'Housing', views.HousingViewSet)
 
urlpatterns = [
    path(r'api/', include(router.urls)),    
    path('', RedirectView.as_view(url="api/")),
]
