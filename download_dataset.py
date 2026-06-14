import os
import json
import csv
import urllib.request
import urllib.error
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import sys

# Configure stdout to use UTF-8 encoding on Windows to support checkmarks and cross symbols
sys.stdout.reconfigure(encoding='utf-8')

# Directory Paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLIANCE_DIR = os.path.join(WORKSPACE_DIR, "compliance_guidelines")
IPO_DIR = os.path.join(WORKSPACE_DIR, "ipo_documents")
JSON_LIST_PATH = os.path.join(WORKSPACE_DIR, "scraped_ipo_list.json")
METADATA_CSV_PATH = os.path.join(WORKSPACE_DIR, "metadata.csv")

# Create Directories if not exist
os.makedirs(COMPLIANCE_DIR, exist_ok=True)
os.makedirs(IPO_DIR, exist_ok=True)

# SEBI Compliance Circulars (Direct, verified PDF download links)
COMPLIANCE_PDFS = {
    "sebi_icdr_master_circular.pdf": "https://www.sebi.gov.in/sebi_data/attachdocs/feb-2026/1770636136785.pdf",
    "sebi_lodr_master_circular.pdf": "https://www.sebi.gov.in/sebi_data/attachdocs/jan-2026/1769776024792.pdf",
    "sebi_sast_master_circular.pdf": "https://avantiscdnprodstorage.blob.core.windows.net/legalupdatedocs/21801/SEBI_Master_Circular_for_Substantial_Acquisition_of_Shares_and_Takeovers_FEB162023.pdf"
}

# Bypass SSL Verification issues for Government/Exchange portals
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def clean_filename(name):
    # Convert name to lowercase and replace spaces/special chars with underscores
    name_clean = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
    name_clean = re.sub(r'\s+', '_', name_clean.strip())
    return name_clean

def is_valid_pdf(file_path):
    """Verify that a file exists and starts with the PDF magic header %PDF."""
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) < 4:
        return False
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False

def download_file(url, target_path, retries=3, timeout=30):
    """Downloads a single file with retries and a custom User-Agent."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response, open(target_path, 'wb') as out_file:
                out_file.write(response.read())
            
            # Check if downloaded file is a valid PDF
            if is_valid_pdf(target_path):
                return True
            else:
                # If not valid PDF, remove it and try again
                if os.path.exists(target_path):
                    os.remove(target_path)
                raise ValueError("Corrupt file: magic number is not %PDF")
                
        except Exception as e:
            # Clean up on failure
            if os.path.exists(target_path):
                os.remove(target_path)
            if attempt == retries:
                print(f"Failed to download {url}: {e} after {retries} attempts.")
                return False
            time.sleep(2) # Wait before retry
    return False

def main():
    print("=" * 60)
    print("STARTING INDIAN IPO AND SEBI COMPLIANCE DATASET CREATOR")
    print("=" * 60)
    
    # 1. Download Compliance Circulars
    print("\n--- Phase 1: Downloading SEBI Compliance Master Circulars ---")
    compliance_results = []
    for filename, url in COMPLIANCE_PDFS.items():
        target_path = os.path.join(COMPLIANCE_DIR, filename)
        print(f"Downloading {filename} from {url}...")
        success = download_file(url, target_path)
        if success:
            file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"✓ Success! Saved to {target_path} ({file_size_mb:.2f} MB)")
            compliance_results.append({
                "type": "Compliance Guideline",
                "name": filename,
                "url": url,
                "local_path": os.path.relpath(target_path, WORKSPACE_DIR),
                "size_mb": round(file_size_mb, 2)
            })
        else:
            print(f"✗ Failed to download compliance file: {filename}")
            
    # 2. Load IPO Documents List
    if not os.path.exists(JSON_LIST_PATH):
        print(f"Error: JSON list not found at {JSON_LIST_PATH}")
        return
        
    with open(JSON_LIST_PATH, "r", encoding="utf-8") as f:
        all_ipo_items = json.load(f)
        
    print(f"\nLoaded {len(all_ipo_items)} unique IPO URLs from json list.")
    
    # We will target at least 100 successful IPO PDF downloads
    target_count = 100
    successful_downloads = []
    failed_items = []
    
    print(f"\n--- Phase 2: Downloading Recent IPO Documents in Parallel (Target: {target_count}) ---")
    
    # Download helper for ThreadPoolExecutor
    def process_item(idx, item):
        comp_name = item["company_name"]
        date = item["date"]
        url = item["pdf_url"]
        
        safe_name = f"{idx}_{clean_filename(comp_name)}_drhp.pdf"
        target_path = os.path.join(IPO_DIR, safe_name)
        
        # Download
        success = download_file(url, target_path)
        if success:
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            return {
                "success": True,
                "company_name": comp_name,
                "date": date,
                "url": url,
                "local_path": os.path.relpath(target_path, WORKSPACE_DIR),
                "size_mb": round(size_mb, 2)
            }
        else:
            return {
                "success": False,
                "company_name": comp_name,
                "url": url
            }

    # Since they are large files, we download with 8 parallel workers
    max_workers = 8
    
    # We will process in chunks or launch all and stop when we hit 100 successful downloads.
    # To keep progress tracking clean, we'll submit all, but we will write a loop.
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, item in enumerate(all_ipo_items, start=1):
            futures.append(executor.submit(process_item, index, item))
            
        completed_count = 0
        success_count = 0
        
        for future in as_completed(futures):
            res = future.result()
            completed_count += 1
            if res["success"]:
                success_count += 1
                successful_downloads.append(res)
                print(f"[{completed_count}/{len(all_ipo_items)}] ✓ Downloaded: {res['company_name']} ({res['size_mb']:.2f} MB)")
            else:
                failed_items.append(res)
                print(f"[{completed_count}/{len(all_ipo_items)}] ✗ Failed: {res['company_name']}")
                
            # If we've successfully reached our target, we can keep going to get a bit more or just finish
            if success_count >= target_count:
                print(f"\nReached target of {target_count} successful IPO PDF downloads!")
                # Cancel pending futures to save bandwidth
                for fut in futures:
                    fut.cancel()
                break

    # 3. Create Metadata CSV
    print("\n--- Phase 3: Writing Dataset Metadata index (metadata.csv) ---")
    
    with open(METADATA_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow(["Index", "Type", "Document Name / Company Name", "Filing Date / Publication", "Source URL", "Local Path", "Size (MB)"])
        
        # Write Compliance Guidelines
        row_idx = 1
        for comp in compliance_results:
            writer.writerow([
                row_idx,
                comp["type"],
                comp["name"],
                "Official Regulation",
                comp["url"],
                comp["local_path"],
                f"{comp['size_mb']} MB"
            ])
            row_idx += 1
            
        # Write IPOs
        for ipo in successful_downloads:
            writer.writerow([
                row_idx,
                "IPO Prospectus",
                ipo["company_name"],
                ipo["date"],
                ipo["url"],
                ipo["local_path"],
                f"{ipo['size_mb']} MB"
            ])
            row_idx += 1
            
    print(f"✓ Metadata index successfully written to {METADATA_CSV_PATH}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Compliance PDFs downloaded: {len(compliance_results)}/3")
    print(f"IPO PDFs downloaded successfully: {len(successful_downloads)}")
    print(f"Total downloaded files in dataset: {len(compliance_results) + len(successful_downloads)}")
    print(f"Failed downloads: {len(failed_items)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
