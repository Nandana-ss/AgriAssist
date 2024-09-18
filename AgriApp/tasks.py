from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from datetime import datetime
from .models import PlantGrowthRecord
from .constants import growth_stage_durations, growth_stages, care_instructions
import logging

logger = logging.getLogger(__name__)

@shared_task
def test_task():
    print("Celery task is working!")

@shared_task
def update_growth_stage_and_send_reminders():
    now = datetime.now()

    # Fetch records from your PlantGrowthRecord model
    records = PlantGrowthRecord.objects.all()

    for record in records:
        if record.last_stage_update:
            days_since_update = (now - record.last_stage_update).days
        else:
            days_since_update = None

        current_stage = record.growth_stage

        if current_stage in growth_stage_durations:
            days_needed = growth_stage_durations[current_stage]

            # Check if it's time to move to the next stage
            if days_since_update is not None and days_since_update >= days_needed:
                # Update the growth stage
                next_stage = growth_stages.get(current_stage, None)
                if next_stage:
                    record.growth_stage = next_stage
                    record.next_growth_stage = growth_stages.get(next_stage, 'No further stages')
                    record.last_stage_update = now
                    record.save()

                    logger.info(f"Updated plant to the next stage: {next_stage} for {record.user.email}")

                    # Prepare the email for the updated stage
                    email_subject = f"Reminder: Your plant has entered the {next_stage} stage"
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

                else:
                    logger.error(f"Next stage for {current_stage} not found in growth_stages.")
                    continue
            else:
                # If the stage doesn't change, still send a reminder
                email_subject = f"Reminder: Your plant is in the {current_stage} stage"
                email_body = f"""
                Hello {record.user.username},

                Your rice plant is currently in the {current_stage} stage.
                Here are your current irrigation and care instructions:

                - Watering: {care_instructions[current_stage].get('Watering', 'No instructions available.')}
                - Fertilizer: {care_instructions[current_stage].get('Fertilizer', 'No instructions available.')}
                - Pesticide: {care_instructions[current_stage].get('Pesticide', 'No instructions available.')}

                Next Stage - {record.next_growth_stage}:
                - Keep following the proper watering and care routine until the next stage.
                """
                logger.info(f"Sending reminder email for current stage: {current_stage} to {record.user.email}")

            # Send the email
            try:
                send_mail(
                    email_subject,
                    email_body,
                    settings.DEFAULT_FROM_EMAIL,
                    [record.user.email],
                    fail_silently=False,
                )
            except Exception as e:
                logger.error(f"Error sending email to {record.user.email}: {e}")
        else:
            logger.error(f"Current stage {current_stage} not found in growth_stage_durations.")