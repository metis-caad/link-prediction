import json
import os
import sys

from django.http.response import HttpResponse
from django.views.generic.edit import CreateView
from .models import Upload
from django.views.decorators.csrf import csrf_exempt


class UploadView(CreateView):
    model = Upload
    fields = ['file']
    basedir = os.path.dirname(os.path.realpath(sys.argv[0]))

    def get_requests_dir(self):
        return self.basedir[:-12] + 'requests/'

    @staticmethod
    def remove_files(path, ext):
        files = os.listdir(path)
        for file in files:
            if file.endswith(ext):
                os.remove(path + '/' + file)

    def cleanup(self):
        requests_dir = self.get_requests_dir()
        for cls in range(1, 6):
            cls_dir_data = requests_dir + 'dataset_clean/' + str(cls)
            self.remove_files(cls_dir_data + '_open', '.graphml')
            self.remove_files(cls_dir_data + '_closed', '.graphml')
            cls_dir_conn = requests_dir + 'connectivity/' + str(cls)
            self.remove_files(cls_dir_conn + '_open', '.json')
            self.remove_files(cls_dir_conn + '_closed', '.json')
        self.remove_files(requests_dir + 'paths/', '.json')

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super(UploadView, self).dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        self.cleanup()
        jsn = json.load(request.FILES['file'])
        requests_dir = self.get_requests_dir()
        for completion_key in jsn:
            completion_obj = jsn[completion_key]
            agraphml_str = completion_obj['completion']
            spatial_class = completion_obj['spatialClass']
            with open(requests_dir + 'dataset_clean/' + spatial_class + '/' + completion_key + '.graphml', 'w') \
                    as agraphml_file:
                agraphml_file.write(agraphml_str)
            connectivity_scores = completion_obj['connectivityScores']
            with open(requests_dir + 'connectivity/' + spatial_class + '/' + completion_key + '.json', 'w') \
                    as connectivity_file:
                connectivity_file.write(json.dumps(connectivity_scores))
            shortest_paths = completion_obj['shortestPaths']
            for corridor_id in shortest_paths:
                sp = shortest_paths[corridor_id]
                with open(requests_dir + 'paths/' + completion_key + '.graphml_' + corridor_id + '.json', 'w') \
                        as paths_file:
                    paths_file.write(json.dumps(sp))
        response = HttpResponse(json.dumps([{'link_prediction_score': 0}]))
        return response
