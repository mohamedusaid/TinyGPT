"""
Fetch clean Science, Math, and Coding texts from Wikipedia and append ~30,000 words to content.txt.
"""

import requests
import tiktoken

OUTPUT_FILE = "content.txt"

TOPICS = [
    ("Computer Science", "Computer_science", 4200),
    ("Python Programming", "Python_(programming_language)", 6000),
    ("Artificial Intelligence", "Artificial_intelligence", 8000),
    ("Mathematics", "History_of_mathematics", 6500),
    ("Planetary Science", "Solar_System", 6000),
]

headers = {"User-Agent": "TinyGPT-STEM/1.0 (mohamedusaid435@gmail.com)"}
enc = tiktoken.get_encoding("gpt2")

def clean_wiki(text):
    for stop in [
        "\n== References ==",
        "\n== See also ==",
        "\n== Further reading ==",
        "\n== External links ==",
        "\n== Notes ==",
        "\n== Bibliography =="
    ]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()

print("Fetching Science, Math, and Coding texts from Wikipedia...")

all_sections = []
total_new_words = 0

for name, title, target in TOPICS:
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true&titles={title}&format=json"
    res = requests.get(url, headers=headers, timeout=15).json()
    pages = res.get("query", {}).get("pages", {})
    for _, p in pages.items():
        raw = p.get("extract", "")
        cleaned = clean_wiki(raw)
        paras = [para.strip() for para in cleaned.split("\n\n") if para.strip()]
        chosen = []
        words_sec = 0
        for para in paras:
            w = len(para.split())
            chosen.append(para)
            words_sec += w
            if words_sec >= target:
                break
        
        section_str = f"=== {name} ===\n\n" + "\n\n".join(chosen)
        all_sections.append(section_str)
        total_new_words += words_sec
        print(f"Added {name}: {words_sec:,} words")

new_stem_text = "\n\n".join(all_sections)
new_words = len(new_stem_text.split())
new_tokens = len(enc.encode(new_stem_text))

# Read original IMDb content.txt
# First, let's keep only the IMDb portion if previous append was run
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    current_content = f.read()

# Separate IMDb reviews from any previous append
if "=== Computer Science ===" in current_content:
    imdb_content = current_content.split("=== Computer Science ===")[0].rstrip()
else:
    imdb_content = current_content.rstrip()

imdb_words = len(imdb_content.split())
imdb_tokens = len(enc.encode(imdb_content))

# Append new clean STEM text
final_content = imdb_content + "\n\n" + new_stem_text

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(final_content)

total_words = len(final_content.split())
total_tokens = len(enc.encode(final_content))

print("-" * 50)
print(f"IMDb reviews (top):          {imdb_words:,} words ({imdb_tokens:,} tokens)")
print(f"STEM text (appended bottom): {new_words:,} words ({new_tokens:,} tokens)")
print(f"Total content.txt:           {total_words:,} words ({total_tokens:,} tokens)")
