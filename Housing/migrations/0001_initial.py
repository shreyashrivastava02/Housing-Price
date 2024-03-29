# Generated by Django 2.1 on 2019-07-17 08:12

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Housing',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ocean_proximity', models.CharField(max_length=50)),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('housing_median_age', models.FloatField()),
                ('total_rooms', models.FloatField()),
                ('total_bedrooms', models.FloatField()),
                ('population', models.FloatField()),
                ('households', models.FloatField()),
                ('median_income', models.FloatField()),
                ('median_house_value', models.FloatField()),
            ],
        ),
    ]
