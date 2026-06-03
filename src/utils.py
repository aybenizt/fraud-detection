import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format for structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
        }
        
        # Capture extra fields passed as `extra` dict, but ensure they don't overwrite standard fields
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            for k, v in record.extra_fields.items():
                if k not in log_record:
                    log_record[k] = v
                    
        return json.dumps(log_record)

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a structured JSON logger.
    """
    logger = logging.getLogger(name)
    # Prevent duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
        
    return logger
