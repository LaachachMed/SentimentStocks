from django.contrib.auth.models import User, Group
from rest_framework import serializers
from .models import TwitterUser, Tweets,AppUser


class AppUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = AppUser
        fields = ['id', 'UserName','Password']

class TwitterUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = TwitterUser
        fields = ['id', 'UserName', 'ApiKey', 'ApiSecretKey', 'AccessToken', 'AccessTokenSecret']


class TweetsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tweets
        fields = ['id', 'keyword', 'date_tweets', 'tweets']
