import sys
from pathlib import Path
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

def main():
    html_path = Path(__file__).parents[1] / "frontend" / "consent.html"
    if not html_path.exists():
        print(f"Error: {html_path} does not exist.")
        sys.exit(1)

    html_content = html_path.read_text(encoding="utf-8")
    
    required_ids = [
        "mock-gateway-overlay",
        "gateway-loader-screen",
        "gateway-success-screen",
        "view-digilocker",
        "view-aa",
        "view-bbps",
        "view-ecommerce",
        "view-udyam",
        "view-merchant",
        "view-kisan",
        "gateway-steps-container"
    ]

    print("Running DOM Verification on consent.html...")
    
    if BeautifulSoup:
        print("Using BeautifulSoup for parsing...")
        soup = BeautifulSoup(html_content, "html.parser")
        errors = 0
        for rid in required_ids:
            elem = soup.find(id=rid)
            if elem is None:
                print(f"  [FAIL] Missing element with ID: #{rid}")
                errors += 1
            else:
                print(f"  [PASS] Found element with ID: #{rid}")
        
        if errors > 0:
            print(f"DOM Verification FAILED with {errors} error(s).")
            sys.exit(1)
    else:
        print("BeautifulSoup not found, falling back to simple string check...")
        errors = 0
        for rid in required_ids:
            # simple string matches since they are unique ids in html
            if f'id="{rid}"' in html_content or f"id='{rid}'" in html_content:
                print(f"  [PASS] Found element with ID: #{rid}")
            else:
                print(f"  [FAIL] Missing element with ID: #{rid}")
                errors += 1
        
        if errors > 0:
            print(f"DOM Verification FAILED with {errors} error(s).")
            sys.exit(1)
            
    print("DOM Verification PASSED successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
