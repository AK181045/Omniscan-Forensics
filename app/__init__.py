import importlib
import sys

# Load the actual FastAPI application package located under 'api/app'
_actual_pkg = importlib.import_module('api.app')

# Replace the current 'app' module entry with the actual package to make "from app.xxx import yyy" work.
sys.modules[__name__] = _actual_pkg
