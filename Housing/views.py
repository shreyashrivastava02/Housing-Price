from django.http.response import HttpResponse
from rest_framework import viewsets
from Housing.models import Housing
from Housing.myserializer import HousingSerializer
# from Housing.models import Housing
from Housing import Hml
from rest_framework.response import Response

class HousingViewSet(viewsets.ModelViewSet):
    queryset = Housing.objects.order_by("-id")
    serializer_class = HousingSerializer
    def create(self, request, *args, **kwargs):
        viewsets.ModelViewSet.create(self, request, *args, **kwargs)
        ob = Housing.objects.latest("id")
        sur = Hml.pred(ob)
        return Response({"status": "Success", "median_house_value": sur, 'tmp': args})