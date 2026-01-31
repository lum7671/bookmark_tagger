import argparse
import datetime
import json
import os
import shutil
import sqlite3
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configuration
DB_FILE = "keyword_cache.db"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class TagDB:
    def __init__(self, db_file=DB_FILE):
        self.conn = sqlite3.connect(db_file)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS page_tags (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    tags TEXT,
                    updated_at TIMESTAMP
                )
            """)

    def get_tags(self, url):
        cursor = self.conn.execute("SELECT tags FROM page_tags WHERE url = ?", (url,))
        row = cursor.fetchone()
        return row[0] if row else None

    def upsert_tags(self, url, title, tags_list):
        tags_str = ",".join(tags_list)
        now = datetime.datetime.now().isoformat()
        with self.conn:
            self.conn.execute("""
                INSERT INTO page_tags (url, title, tags, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    tags = excluded.tags,
                    updated_at = excluded.updated_at
            """, (url, title, tags_str, now))

    def close(self):
        self.conn.close()

def setup_gemini():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

def get_page_content(url):
    try:
        response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            text = " ".join([p.get_text() for p in soup.find_all('p')[:5]])
            return title, text[:1000]
    except Exception:
        pass
    return "", ""

def generate_tags(client, title, content):
    if not client:
        return []
    
    prompt = f"""
    Analyze the following webpage content and suggest 3-5 relevant tags for a bookmark.
    Response MUST be a comma-separated list of tags only. No explanation.
    
    Title: {title}
    Content Snippet: {content}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="text/plain" 
            )
        )
        return [tag.strip() for tag in response.text.split(',') if tag.strip()]
    except Exception as e:
        # Check for quota error patterns
        err_msg = str(e).lower()
        if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg:
             print("Warning: Gemini API Quota Exceeded!")
             raise BlockingIOError("API Quota Reached") # Signal to stop
        print(f"Gemini API Error: {e}")
        return []

def collect_bookmarks(node, bucket):
    if node.get('type') == 'text/x-moz-place':
        bucket.append(node)
    
    if 'children' in node:
        for child in node['children']:
            collect_bookmarks(child, bucket)

def process_phase1(bookmarks, db, client, init_mode):
    print(f"Phase 1: Syncing DB and Generating Tags (Total: {len(bookmarks)})")
    api_quota_reached = False

    for i, node in enumerate(bookmarks):
        url = node.get('uri', '')
        title = node.get('title', '')
        if not url.startswith('http'):
            continue

        existing_tags = node.get('tags', '').split(',')
        existing_tags = [t.strip() for t in existing_tags if t.strip()]

        # 1. Check if we should use existing file tags
        # If NOT init_mode and we have manual tags, trust them and save to DB.
        if not init_mode and existing_tags:
            # Sync manual tags to DB
            db.upsert_tags(url, title, existing_tags)
            continue # Done for this node

        # 2. Check DB Cache
        cached_tags = db.get_tags(url)
        if not init_mode and cached_tags is not None:
            # Cache hit, and we are not forced to refresh
            continue 

        # 3. AI Generation (init_mode OR (no tags AND no cache))
        if api_quota_reached:
            # Skip subsequent AI calls, but keep processing existing tags sync
            continue

        print(f"[{i+1}/{len(bookmarks)}] AI Analysis for: {title[:40]}...")
        page_title, content = get_page_content(url)
        # Use page title if bookmark title is empty
        final_title = title if title else page_title

        if final_title or content:
            try:
                new_tags = generate_tags(client, final_title, content)
                if new_tags:
                    print(f"  -> Generated: {new_tags}")
                    # If init_mode, we replace. If normal, we merge with existing empty? 
                    # Logic says we only get here if existing was empty (normal mode).
                    # In init mode, we overwrite.
                    db.upsert_tags(url, final_title, new_tags)
                else:
                    print("  -> No tags suggested.")
                    # Mark as visited in DB with empty tags to avoid re-scan?
                    # db.upsert_tags(url, final_title, []) 
            except BlockingIOError:
                print("!!! API Limit Reached. Stopping AI generation for this session. !!!")
                api_quota_reached = True
        else:
            print("  -> Content fetch failed.")
        
        time.sleep(1) # Graceful delay

def process_phase2(node, db):
    # Recursively update JSON tree from DB
    if node.get('type') == 'text/x-moz-place':
        url = node.get('uri', '')
        if url.startswith('http'):
            cached_tags_val = db.get_tags(url)
            if cached_tags_val:
                # Merge logic? Or Request said: 
                # "output file to existing file, adding tags"
                # "If init, delete existing data and update" -> Implies overwrite in DB.
                # Here we just reflect DB to JSON. 
                # If cached_tags_val exists, it's the source of truth.
                node['tags'] = cached_tags_val
    
    if 'children' in node:
        for child in node['children']:
            process_phase2(child, db)

def main():
    parser = argparse.ArgumentParser(description="Auto-tag bookmarks using Gemini AI")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("--init", action="store_true", help="Force regenerate all tags using AI, ignoring existing tags/cache.")
    args = parser.parse_args()

    input_path = args.input_file
    temp_path = input_path + ".temp"

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    # Backup / Work on temp
    shutil.copy2(input_path, temp_path)
    print(f"Working on copy: {temp_path}")

    # Load Data
    with open(temp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten for iteration
    all_bookmarks_list = []
    collect_bookmarks(data, all_bookmarks_list)

    # Setup Resources
    db = TagDB()
    client = setup_gemini()
    
    if not client:
        return

    # Phase 1: Enrich DB
    process_phase1(all_bookmarks_list, db, client, args.init)

    # Phase 2: Update JSON from DB
    print("Phase 2: Updating JSON from Database...")
    process_phase2(data, db)

    # Write Output (Overwrite original input file as requested)
    # User said: "Output file same as existing file name" (a.json)
    print(f"Saving finalized bookmarks to {input_path}...")
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Cleanup
    db.close()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print("Optimization Complete.")

if __name__ == "__main__":
    main()
