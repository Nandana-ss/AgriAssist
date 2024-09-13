from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    # Fields to display in the list view
    list_display = ('email', 'first_name', 'last_name')
    
    # Fields to use in the search functionality
    search_fields = ('email', 'first_name', 'last_name')
    
    # Ordering of the list
    ordering = ('email',)
    
    # Fields to show in the form view (you can customize this as needed)
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'password1', 'password2'),
        }),
    )
# Register the custom user admin
admin.site.register(CustomUser, CustomUserAdmin)


# plant growth tracking
from .models import PlantGrowthRecord

class PlantGrowthRecordAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'growth_stage',
        'date_uploaded',
        'soil_type',
        'water_frequency',
        'fertilizer_type',
    )
    list_filter = (
        'growth_stage',
        'date_uploaded',
        'soil_type',
    )
    search_fields = ('user__username', 'growth_stage', 'soil_type')

admin.site.register(PlantGrowthRecord, PlantGrowthRecordAdmin)