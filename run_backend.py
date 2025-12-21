#!/usr/bin/env python3
"""
Backend runner script for local development.
Resolves ModuleNotFoundError by ensuring the backend directory is in the Python path.

Usage:
    python run_backend.py
    # or
    ./run_backend.py
"""

import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_dir)

# Change to backend directory for .env file resolution
os.chdir(backend_dir)

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[backend_dir]
    )
