from django.db import models
import os


def upload_to(instance, filename):
    """Generate upload path for detection images."""
    return os.path.join('uploads', filename)


class DetectionResult(models.Model):
    """Store detection results in the database."""
    image = models.ImageField(upload_to=upload_to)
    is_weapon = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    predicted_label = models.CharField(max_length=50, default='Unknown')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Detection Result'
        verbose_name_plural = 'Detection Results'

    def __str__(self):
        status = "WEAPON DETECTED" if self.is_weapon else "No Weapon"
        return f"{status} - {self.confidence:.1f}% - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

