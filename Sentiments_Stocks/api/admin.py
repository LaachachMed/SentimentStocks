from django.contrib import admin
from .models import TwitterUser,Tweets,AppUser

# Register my models here
admin.site.site_header = "SentiStocks Administration"
admin.site.register(AppUser)
admin.site.register(TwitterUser)
admin.site.register(Tweets)