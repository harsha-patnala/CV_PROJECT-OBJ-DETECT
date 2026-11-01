# conftest.py — pytest configuration
import sys
import os

# Add project root to sys.path so tests can import modules
sys.path.insert(0, os.path.dirname(__file__))
