import os
import requests
import time

# ── Corrected book IDs ───────────────────────────────────────
# Fix: Novum Organum is 45988, not 4598
eras = {
    "Renaissance":   [1515, 15272, 45988, 779],   # 45988 = Novum Organum (fixed)
    "Enlightenment": [829, 521, 3300, 147],
    "Romantic":      [1342, 84, 9622, 82],
    "Victorian":     [1400, 174, 1260, 4300]       # 1260 = Jane Eyre (fixed)
}

# ── Two URL patterns Gutenberg uses ─────────────────────────
def get_urls(book_id):
    """Return list of URLs to try, in order of preference."""
    return [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

# ── Main downloader ──────────────────────────────────────────
def download_gutenberg_books():
    failed = []   # keep track of any books that couldn't download

    for era, ids in eras.items():

        # Create a folder for each era if it doesn't exist
        if not os.path.exists(era):
            os.makedirs(era)
            print(f"Created folder: {era}/")

        for book_id in ids:
            filename = os.path.join(era, f"{book_id}.txt")

            # Skip if already downloaded (useful if you re-run the script)
            if os.path.exists(filename):
                print(f"  Already exists, skipping: {filename}")
                continue

            # Try each URL pattern until one works
            downloaded = False
            for url in get_urls(book_id):
                print(f"  Trying: {url}")
                try:
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
                            f.write(response.text)
                        size_kb = os.path.getsize(filename) // 1024
                        print(f"  Saved: {filename} ({size_kb} KB)")
                        downloaded = True
                        break   # stop trying other URLs
                except requests.exceptions.RequestException as e:
                    print(f"  Connection error: {e}")

            if not downloaded:
                print(f"  FAILED: Could not download ID {book_id} for {era}")
                failed.append((era, book_id))

            # Small delay — be polite to Gutenberg's servers
            time.sleep(1)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    total = sum(len(ids) for ids in eras.values())
    print(f"Total books attempted : {total}")
    print(f"Successfully downloaded: {total - len(failed)}")

    if failed:
        print(f"\nFailed downloads ({len(failed)}):")
        for era, book_id in failed:
            print(f"  Era: {era}  |  ID: {book_id}")
        print("\nFor failed books, manually download from:")
        print("https://www.gutenberg.org/ebooks/<ID>")
    else:
        print("\nAll books downloaded successfully!")

if __name__ == "__main__":
    download_gutenberg_books()