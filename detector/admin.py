from django.contrib import admin
from .models import DetectionResult


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ('predicted_label', 'is_weapon', 'confidence', 'created_at')
    list_filter = ('is_weapon', 'created_at')
    search_fields = ('predicted_label',)
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)
