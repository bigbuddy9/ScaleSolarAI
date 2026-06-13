"""
Diagnostic: embed a test query and call the match_knowledge_chunks RPC,
print what comes back. Confirms the search pipeline works outside n8n.
"""
import os
from openai import AzureOpenAI
from supabase import create_client

SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_KEY     = os.environ["SUPABASE_KEY"]
AZURE_ENDPOINT   = os.environ["AZURE_ENDPOINT"]
AZURE_KEY        = os.environ["AZURE_KEY"]
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")

az = AzureOpenAI(api_key=AZURE_KEY, azure_endpoint=AZURE_ENDPOINT, api_version="2024-02-01")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

query = "how do I prevent the i want to think about it objection"
print(f"Query: {query}\n")

emb = az.embeddings.create(input=[query], model=EMBED_DEPLOYMENT).data[0].embedding
print(f"Embedding length: {len(emb)}\n")

for threshold in (0.5, 0.3, 0.1):
    res = sb.rpc("match_knowledge_chunks", {
        "query_embedding": emb,
        "match_threshold": threshold,
        "match_count": 4,
    }).execute()
    print(f"--- threshold {threshold}: {len(res.data)} matches ---")
    for r in res.data:
        print(f"  [{r['similarity']:.3f}] {r['source']}: {r['content'][:90]}...")
    print()
