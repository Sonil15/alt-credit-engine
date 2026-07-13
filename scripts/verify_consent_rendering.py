import sys
import subprocess
import time
import urllib.request
import urllib.error
import os

def main():
    print("Running Consent Rendering verification...")
    env = os.environ.copy()
    env["USE_SQLITE"] = "true"
    env["AUTO_SEED_ON_STARTUP"] = "true"
    env["SIMULATE_ALL_FACETS"] = "true"
    
    # Path to python in virtual env
    if sys.platform == "win32":
        uvicorn_path = os.path.join(".venv", "Scripts", "uvicorn.exe")
    else:
        uvicorn_path = os.path.join(".venv", "bin", "uvicorn")

    if not os.path.exists(uvicorn_path):
        print(f"Error: {uvicorn_path} not found.")
        sys.exit(1)

    print(f"Starting server with: {uvicorn_path} api.main:app --port 8000")
    
    process = None
    try:
        process = subprocess.Popen(
            [uvicorn_path, "api.main:app", "--port", "8000"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to spin up
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Verify it started or is running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("Server failed to start immediately.")
            print(f"Stdout: {stdout}")
            print(f"Stderr: {stderr}")
            sys.exit(1)

        # Test request
        url = "http://localhost:8000/consent"
        print(f"Sending GET request to {url}...")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                status = response.status
                html = response.read().decode("utf-8")
                
                print(f"Response Status: {status}")
                if status == 200:
                    print("  [PASS] Request returned 200 OK")
                    # Check if our new components are present in the rendered output
                    if "mock-gateway-overlay" in html:
                        print("  [PASS] Found mock-gateway-overlay in rendered HTML")
                        print("Consent Page rendered successfully!")
                    else:
                        print("  [FAIL] Rendered HTML missing mock-gateway-overlay")
                        sys.exit(1)
                else:
                    print(f"  [FAIL] Request returned status {status}")
                    sys.exit(1)
        except urllib.error.URLError as e:
            print(f"  [FAIL] Connection failed: {e}")
            sys.exit(1)
            
    finally:
        if process:
            print("Terminating server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, killing it...")
                process.kill()
                process.wait()
            print("Server stopped.")

if __name__ == "__main__":
    main()
