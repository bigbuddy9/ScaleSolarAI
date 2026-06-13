"""
Diagnostic: replicate the n8n chat completion call exactly - embed a query,
search, build the same payload, POST to the Azure chat deployment, print the
raw response so we see what Azure actually complains about.
"""
import os, json, requests
from openai import AzureOpenAI
from supabase import create_client

SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_KEY     = os.environ["SUPABASE_KEY"]
AZURE_ENDPOINT   = os.environ["AZURE_ENDPOINT"].rstrip("/")
AZURE_KEY        = os.environ["AZURE_KEY"]
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")
CHAT_DEPLOYMENT  = os.getenv("CHAT_DEPLOYMENT", "gpt-5.4")
CHAT_API_VER     = os.getenv("CHAT_API_VER", "2024-12-01-preview")

az = AzureOpenAI(api_key=AZURE_KEY, azure_endpoint=AZURE_ENDPOINT, api_version="2024-02-01")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

query = "how do I prevent the i want to think about it objection"
emb = az.embeddings.create(input=[query], model=EMBED_DEPLOYMENT).data[0].embedding
res = sb.rpc("match_knowledge_chunks", {
    "query_embedding": emb, "match_threshold": 0.3, "match_count": 4,
}).execute()
context = "\n\n---\n\n".join(r["content"] for r in res.data)
print(f"Context chars: {len(context)}\n")

url = f"{AZURE_ENDPOINT}/openai/deployments/{CHAT_DEPLOYMENT}/chat/completions?api-version={CHAT_API_VER}"
print(f"URL: {url}\n")

payload = {
    "messages": [
        {"role": "system", "content": "You are the Scale Solar AI coaching assistant. Only use the context below.\n\nCONTEXT:\n" + context},
        {"role": "user", "content": query},
    ],
    "max_completion_tokens": 500,
    "temperature": 0.7,
}

r = requests.post(url, headers={"api-key": AZURE_KEY, "Content-Type": "application/json"}, json=payload)
print(f"--- WITH temperature: HTTP {r.status_code} ---")
print(r.text[:1500])
print()

del payload["temperature"]
r2 = requests.post(url, headers={"api-key": AZURE_KEY, "Content-Type": "application/json"}, json=payload)
print(f"--- WITHOUT temperature: HTTP {r2.status_code} ---")
print(r2.text[:1500])
