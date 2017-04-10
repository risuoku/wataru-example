import os

APPLICATION_ENV = os.environ.get('APPLICATION_ENV')



# environment dependent settings

if APPLICATION_ENV == 'docker':
    from .docker import *
else:
    from .local import *
