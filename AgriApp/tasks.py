from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from datetime import datetime
from .models import PlantGrowthRecord
from .constants import growth_stage_durations, growth_stages, care_instructions

@shared_task
def update_growth_stage_and_send_reminders():
    now = datetime.now()

    # Fetch records from PlantGrowthRecord model
    records = PlantGrowthRecord.objects.all()

    for record in records:
        days_since_update = (now - record.last_stage_update).days
        current_stage = record.growth_stage

        # Ensure the current stage exists in the growth_stage_durations
        if current_stage in growth_stage_durations:
            days_needed = growth_stage_durations[current_stage]

            # Check if it's time to move to the next stage
            if days_since_update >= days_needed:
                # Determine the next growth stage
                next_stage = growth_stages.get(current_stage, None)
                if next_stage:
                    # Update the record's growth stage and next growth stage
                    record.growth_stage = next_stage
                    record.next_growth_stage = growth_stages.get(next_stage, 'No further stages')
                    record.last_stage_update = now
                    record.save()

                    # Fetch care instructions
                    watering_instructions = care_instructions.get(next_stage, {}).get('Watering', 'No instructions available.')
                    fertilizer_instructions = care_instructions.get(next_stage, {}).get('Fertilizer', 'No instructions available.')
                    pesticide_instructions = care_instructions.get(next_stage, {}).get('Pesticide', 'No instructions available.')

                    # Prepare and send the email
                    email_subject = f"Irrigation Reminder: {next_stage} Stage for Your Plant"
                    email_body = f"""
                    Hello {record.user.username},

                    Your rice plant has moved to the {next_stage} stage.
                    Here are your updated care instructions:

                    - **Watering**: {watering_instructions}
                    - **Fertilizer**: {fertilizer_instructions}
                    - **Pesticide**: {pesticide_instructions}

                    Next Stage - {record.next_growth_stage}:
                    - **Watering**: Please ensure you follow the recommended schedule for the next stage.

                    Thank you for using AgriAssist!
                    """

                    try:
                        send_mail(
                            email_subject,
                            email_body,
                            settings.DEFAULT_FROM_EMAIL,
                            [record.user.email],
                            fail_silently=False,
                        )
                    except Exception as e:
                        # Log or print error message
                        print(f"Error sending email to {record.user.email}: {e}")
