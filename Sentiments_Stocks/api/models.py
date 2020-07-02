from django.db import models


# Create your models here.

# Admin of the application will be saved by this
class AppUser(models.Model):
    UserName = models.CharField(max_length=128)
    Password = models.CharField(max_length=256)

# to save a Twitter Developer Account
class TwitterUser(models.Model):
    UserName = models.CharField(max_length=256)
    ApiKey = models.CharField(max_length=256)
    ApiSecretKey = models.CharField(max_length=256)
    AccessToken = models.CharField(max_length=256)
    AccessTokenSecret = models.CharField(max_length=256)

# to save the extracted tweets
class Tweets(models.Model):
    keyword = models.CharField(max_length=256)
    tweets = models.CharField(max_length=512)
    date_tweets = models.DateTimeField(max_length=256)
