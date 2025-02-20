import json


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    try:
        if is_json_serializable(obj):
            return obj
        else:
            if isinstance(obj, dict):
                return {convert_json(k): convert_json(v) for k, v in obj.items()}

            elif isinstance(obj, tuple):
                return (convert_json(x) for x in obj)

            elif isinstance(obj, list):
                return [convert_json(x) for x in obj]

            elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
                return convert_json(obj.__name__)

            elif hasattr(obj, "__dict__") and obj.__dict__:
                obj_dict = {
                    convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
                }
                return {str(obj): obj_dict}

            return str(obj)
    except Exception as e:
        print(f"Error converting object to JSON: {e}")
        return None


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False
