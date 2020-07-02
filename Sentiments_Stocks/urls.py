"""Sentiments_Stocks URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import include, path
from rest_framework import routers
from Sentiments_Stocks.api import views

router = routers.DefaultRouter()
router.register(r'twitteruser', views.TwitterUserViewSet)
router.register(r'tweets', views.TweetsViewSet)
router.register(r'appuser', views.AppUserViewSet)


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', admin.site.urls),
    #path('admin/', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path(r'gettweets/<keyword>/', views.GetTweets.gettweets),
    path(r'predict_sentiments/<model_name>/<tweets>/', views.PredictSentiment.predict_sentiments),
    path(r'upload_file_test/<file>/', views.upload_file_test),
    path(r'upload_file_test/', views.upload_file_test),
    path(r'getsentimentstweets/', views.getsentimentstweets),
    path(r'stock_values/<stockname>/', views.stock_values),
    path(r'calculate_sentiment_score/', views.calculate_sentiment_score),
    path(r'select_users/', views.select_users),
    path(r'chart_sentiment_analysis/<keyword>/<int:period>/', views.chart_sentiment_analysis),
    path(r'maps_sentiment/<keyword>/', views.map_sentiment)
]