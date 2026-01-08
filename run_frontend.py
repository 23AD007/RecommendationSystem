#!/usr/bin/env python3
"""
Run the Streamlit frontend for the Packaging Recommendation System
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    try:
        # Change to the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_root)

        # Path to the Streamlit app
        app_path = os.path.join("src", "frontend", "app.py")

        # Run Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", "8501"]
        print("Starting Streamlit frontend...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nFrontend stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()