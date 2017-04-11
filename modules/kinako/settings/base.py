import os

APPLICATION_ENV = os.environ.get('RIGEL_ENV')
APPLICATION_MODULE_ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(os.path.dirname(__file__))
    )
)
