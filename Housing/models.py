from django.db import models

class Housing(models.Model):
    # longitude    latitude    housing_median_age    total_rooms    total_bedrooms    population    households    median_income    median_house_value    ocean_proximity

    ocean_proximity = models.CharField(max_length=50)
    latitude = models.FloatField()
    longitude = models.FloatField()
    housing_median_age = models.FloatField()
    total_rooms = models.FloatField()
    total_bedrooms = models.FloatField()
    population = models.FloatField()
    households = models.FloatField()
    median_income = models.FloatField()
    median_house_value = models.FloatField()
    
    def to_dict(self):
        return {
            'ocean_proximity':self.ocean_proximity ,
            'latitude':self.latitude ,
            'longitude':self.longitude ,
            'housing_median_age':self.housing_median_age ,
            'total_rooms':self.total_rooms ,
            'total_bedrooms':self.total_bedrooms ,
            'population':self.population ,
            'households':self.households ,
            'median_income':self.median_income ,
            'median_house_value':self.median_house_value ,
                      
        }                                          
