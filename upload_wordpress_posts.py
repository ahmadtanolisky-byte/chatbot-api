import requests, os, re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if not exists
if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Replace this with your actual WordPress website URL
WORDPRESS_SITE = "https://www.skymarketing.com.pk"

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]

def fetch_wordpress_data(endpoint):
    """Fetch paginated posts or pages from the WP REST API"""
    items = []
    page = 1
    while True:
        url = f"{WORDPRESS_SITE}/wp-json/wp/v2/{endpoint}?per_page=100&page={page}"
        r = requests.get(url)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        items.extend(data)
        page += 1
    return items

def upload_to_pinecone(items, content_type="post"):
    for item in items:
        title = item["title"]["rendered"]
        content = clean_html(item["content"]["rendered"])
        full_text = f"{title}\n\n{content}"
        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            embed = client.embeddings.create(model="text-embedding-3-small", input=chunk)
            vector = embed.data[0].embedding
            index.upsert([{
                "id": f"{content_type}_{item['id']}_chunk_{i}",
                "values": vector,
                "metadata": {
                    "type": content_type,
                    "title": title,
                    "url": item["link"],
                    "text": chunk
                }
            }])
        print(f"✅ Uploaded {content_type}: {title}")

# Fetch posts and pages
posts = fetch_wordpress_data("posts")
pages = fetch_wordpress_data("pages")

print(f"📰 Found {len(posts)} posts and {len(pages)} pages")

# Upload both to Pinecone
upload_to_pinecone(posts, "post")
upload_to_pinecone(pages, "page")

print("🎉 Done uploading all WordPress content to Pinecone!")
