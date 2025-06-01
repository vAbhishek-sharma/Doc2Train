# utils/error_handler.py - NEW FILE
import functools
import traceback
from typing import Callable, Any

def safe_execute(default_return=None, log_errors=True):
    """Decorator to safely execute functions with error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    print(f"❌ Error in {func.__name__}: {e}")
                    if hasattr(args[0], 'config') and args[0].config.get('verbose'):
                        traceback.print_exc()
                return default_return
        return wrapper
    return decorator

class ErrorBoundary:
    """Context manager for error boundaries"""
    def __init__(self, operation_name: str, continue_on_error: bool = True):
        self.operation_name = operation_name
        self.continue_on_error = continue_on_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"❌ Error in {self.operation_name}: {exc_val}")
            return self.continue_on_error  # Suppress exception if continue_on_error
