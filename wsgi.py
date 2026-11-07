from app import app, init_mongodb, load_models


init_mongodb()
if not load_models():
    raise RuntimeError("Failed to load the required classification and segmentation models")
