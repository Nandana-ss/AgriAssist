# myproject/celery.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Set default Django settings for Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AgriAssist.settings')

app = Celery('AgriAssist')

# Load task modules from all registered Django app configs.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Autodiscover tasks in all apps
app.autodiscover_tasks()
