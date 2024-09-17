
from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from datetime import datetime, timedelta
from .models import PlantGrowthRecord
from .constants import growth_stage_durations  # Define this in a constants file
from .constants import growth_stages
from .constants import care_instructions
@shared_task
def update_growth_stage_and_send_reminders():
    now = datetime.now()

    # Fetch records from your PlantGrowthRecord model
    records = PlantGrowthRecord.objects.all()

    for record in records:
        days_since_update = (now - record.last_stage_update).days
        current_stage = record.growth_stage
        if current_stage in growth_stage_durations:
            days_needed = growth_stage_durations[current_stage]

            # Check if it's time to move to the next stage
            if days_since_update >= days_needed:
                # Update the growth stage
                next_stage = growth_stages.get(current_stage, None)
                if next_stage:
                    record.growth_stage = next_stage
                    record.next_growth_stage = growth_stages.get(next_stage, 'No further stages')
                    record.last_stage_update = now
                    record.save()

                    # Send reminder email
                    email_subject = f"Irrigation Reminder for {next_stage}"
                    email_body = f"""
                    Hello {record.user.username},

                    Your rice plant has moved to the {next_stage} stage.
                    Here are your updated irrigation and care instructions:

                    - Watering: {care_instructions[next_stage].get('Watering', 'No instructions available.')}
                    - Fertilizer: {care_instructions[next_stage].get('Fertilizer', 'No instructions available.')}
                    - Pesticide: {care_instructions[next_stage].get('Pesticide', 'No instructions available.')}

                    Next Stage - {record.next_growth_stage}:
                    - Watering: Please make sure to follow the next watering schedule.
                    """

                    send_mail(
                        email_subject,
                        email_body,
                        settings.DEFAULT_FROM_EMAIL,
                        [record.user.email],
                        fail_silently=False,
                    )
