from django.db import models


class Upload(models.Model):
    file = models.FileField()
    date = models.DateTimeField(auto_now_add=True)
