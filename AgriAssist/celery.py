from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.conf import settings
from celery.schedules import crontab
# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AgriAssist.settings')

app = Celery('AgriAssist')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related config keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# Beat schedule configuration
app.conf.beat_schedule = {
    'update-growth-stage-and-send-reminders-every-day': {
        'task': 'AgriApp.tasks.update_growth_stage_and_send_reminders',
        'schedule': crontab(hour=0, minute=0),  # Every day at midnight
    },
}

# You can also add this to handle timezone
app.conf.timezone = 'UTC'
