from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),
    path('crop_recommendation/', login_required(views.crop_recommendation), name='crop_recommendation'),
    path('progress/', login_required(views.progress), name='progress'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('predict_price/', views.predict_price, name='predict_price'),
    path('disease_recognition/', views.disease_recognition, name='disease_recognition'),
    path('show_login/', views.show_login, name='show_login'),
    path('user_login/', views.user_login, name='user_login'),
    path('register/', views.register, name='register'),
    path('logout_user/', views.logout_user, name='logout'),
    path('predict_plant_growth/', views.predict_plant_growth, name='predict_plant_growth'),
    path('fetch_user_history/', views.fetch_user_history, name='fetch_user_history'),
    path('profile/', views.profile, name='profile'),
]