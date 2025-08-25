from django.db import models

# Create your models here.
class login(models.Model):
    username=models.CharField(max_length=100)
    password=models.CharField(max_length=100)
    usertype=models.CharField(max_length=100)


class authority(models.Model):
    name=models.CharField(max_length=100)
    email=models.CharField(max_length=100)
    phone=models.CharField(max_length=100)
    gender=models.CharField(max_length=100)
    photo=models.CharField(max_length=100)
    LOGIN=models.ForeignKey(login, on_delete=models.CASCADE)


class places(models.Model):
    place_name = models.CharField(max_length=100)
    district = models.CharField(max_length=100)


class allocation(models.Model):
    AUTHORITY = models.ForeignKey(authority, on_delete=models.CASCADE)
    PLACE = models.ForeignKey(places, on_delete=models.CASCADE)


class complaints(models.Model):
    date = models.CharField(max_length=100)
    complaint = models.CharField(max_length=100)
    reply = models.CharField(max_length=200)
    AUTHORITY = models.ForeignKey(authority, on_delete=models.CASCADE)


class suggestions(models.Model):
    date = models.CharField(max_length=100)
    time = models.CharField(max_length=100)
    suggestion = models.CharField(max_length=200)
    AUTHORITY = models.ForeignKey(authority, on_delete=models.CASCADE)


class camera(models.Model):
    cam_model = models.CharField(max_length=100)
    manufacturer = models.CharField(max_length=100)
    area = models.CharField(max_length=100)
    PLACE = models.ForeignKey(places, on_delete=models.CASCADE)


class detections(models.Model):
    date = models.CharField(max_length=100)
    time = models.CharField(max_length=100)
    image = models.CharField(max_length=200)
    message = models.CharField(max_length=200, default="")
    CAMERA = models.ForeignKey(camera, on_delete=models.CASCADE)
