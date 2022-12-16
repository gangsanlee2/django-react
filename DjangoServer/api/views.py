from rest_framework.response import Response
from rest_framework.decorators import api_view
import datetime

@api_view(['GET'])
def hello(request):
    print('##################################################')
    print(f'## Server Started At {datetime.datetime.now()} ##')
    print('##################################################')
    return Response({'message': "Server Started !"})