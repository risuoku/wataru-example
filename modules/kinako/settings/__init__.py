import os

if os.environ.get('APPLICATION_ENV') == 'docker':
    from .docker import *
else:
    from .local import *
