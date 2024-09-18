from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.utils import timezone
# register
class CustomUser(AbstractUser):
    phone = models.CharField(max_length=15, unique=True, null=True, blank=True)

    groups = models.ManyToManyField(
        Group,
        related_name='customuser_set',  # Custom related name for CustomUser
        blank=True,
        help_text='The groups this user belongs to.',
        verbose_name='groups',
    )

    user_permissions = models.ManyToManyField(
        Permission,
        related_name='customuser_set',  # Custom related name for CustomUser
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions',
    )

class PlantGrowthRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='growth_images/')
    growth_stage = models.CharField(max_length=50)
    watering_instructions = models.TextField()
    fertilizer_instructions = models.TextField()
    pesticide_instructions = models.TextField()
    next_growth_stage = models.CharField(max_length=50, blank=True, null=True)
    next_watering_instructions = models.TextField(blank=True, null=True)
    next_fertilizer_instructions = models.TextField(blank=True, null=True)
    next_pesticide_instructions = models.TextField(blank=True, null=True)
    date_uploaded = models.DateTimeField(auto_now_add=True)
    soil_type = models.CharField(max_length=100)
    water_frequency = models.CharField(max_length=100)
    fertilizer_type = models.CharField(max_length=100)
    additional_details = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_stage_update = models.DateTimeField(default=timezone.now)

def __str__(self):
    return f"Growth Stage: {self.growth_stage} - Uploaded on: {self.date_uploaded}"

class Review(models.Model):
    record = models.OneToOneField(PlantGrowthRecord, on_delete=models.CASCADE)
    review_text = models.TextField()
    rating = models.IntegerField() 
    reviewed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review for {self.record} - {self.rating} stars"

