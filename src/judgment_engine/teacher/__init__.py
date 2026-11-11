"""教師アダプタ。"""

from .base import LabelResult, SampleRequest, TeacherAdapter, TeacherError
from .openai_compatible import OpenAICompatibleTeacher

__all__ = ["LabelResult", "SampleRequest", "TeacherAdapter", "TeacherError", "OpenAICompatibleTeacher"]
