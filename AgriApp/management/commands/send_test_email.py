from django.core.management.base import BaseCommand
from django.core.mail import send_mail
from django.conf import settings

class Command(BaseCommand):
    help = 'Send a test email to verify email configuration.'

    def handle(self, *args, **options):
        subject = 'Test Email from Django'
        message = 'This is a test email sent from Django to verify email settings.'
        recipient_list = ['ssnandana200@gmail.com']  # Replace with a valid email address

        try:
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                recipient_list,
                fail_silently=False,
            )
            self.stdout.write(self.style.SUCCESS('Test email sent successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error sending test email: {e}'))
