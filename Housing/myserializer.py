'''
Created on Jul 17, 2019

@author: shreya0008
'''
from Housing.models import Housing
from rest_framework import serializers

class HousingSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Housing
        fields = "__all__"

