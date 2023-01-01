import json

from django.http.response import HttpResponse
from django.views.generic.edit import CreateView
from .models import Upload
from django.views.decorators.csrf import csrf_exempt


class UploadView(CreateView):
    model = Upload
    fields = ['file']

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super(UploadView, self).dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        print(json.load(request.FILES['file']))
        response = HttpResponse(json.dumps([{'link_prediction_score': 0}]))
        return response
