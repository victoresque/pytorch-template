
{
    "version": 1, 
    "disable_existing_loggers": false, 
    "formatters": {
        "simple": {"format": "%(message)s"}, 
        "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    }, 
    "handlers": {
        "console": {
            "class": "logging.StreamHandler", 
            "level": "DEBUG", 
            "formatter": "simple", 
            "stream": "ext://sys.stdout"
            }, 
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler", 
            "level": "INFO", 
            "formatter": "datetime", 
            "filename": "info.log", 
            "maxBytes": 10485760, 
            "backupCount": 20, "encoding": "utf8"
        }
    }, 
    "root": {
        "level": "INFO", 
        "handlers": [
            "console", 
            "info_file_handler"
        ]
    }
}