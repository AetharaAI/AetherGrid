I need you to replace OpenAI embeddings in AetherGrid with a dual-provider system:

PRIMARY: Local embeddings using sentence-transformers (for privacy & cost)
FALLBACK: Cohere embeddings API (for quality when needed)

REQUIREMENTS:

1. REMOVE all OpenAI dependencies completely
   - Remove from pyproject.toml and requirements.txt
   - Remove OPENAI_API_KEY from .env.example
   - No OpenAI code anywhere

2. IMPLEMENT local embeddings as primary provider
   - Use sentence-transformers library
   - Default model: "all-mpnet-base-v2" (768 dimensions)
   - Should work offline, no API calls
   - Batch processing for efficiency
   - Store model in ~/.cache (HuggingFace default)

3. ADD Cohere as optional fallback/alternative
   - Use cohere-python SDK
   - Support both embed-english-v3.0 and embed-multilingual-v3.0
   - Make it configurable via environment variable
   - COHERE_API_KEY in .env.example (optional)

4. CREATE a unified EmbeddingProvider interface that:
   - Abstracts the underlying provider (local vs Cohere)
   - Allows runtime switching via config
   - Has these methods:
     * generate_embedding(text: str) -> List[float]
     * generate_embeddings_batch(texts: List[str]) -> List[List[float]]
     * get_dimensions() -> int
   - Handles errors gracefully (fallback to alternate provider if one fails)

5. UPDATE Weaviate schema to support 768 dimensions (mpnet-base-v2)
   - Make it configurable if using different models
   - Document dimension mapping in README

6. UPDATE the processing worker (src/processing/embedding_worker.py)
   - Use the new EmbeddingProvider interface
   - Default to local embeddings
   - Log which provider is being used

7. UPDATE docker-compose.yml and documentation
   - Remove OpenAI references
   - Add Cohere as optional
   - Update Quick Start to work WITHOUT any API keys (local-first)

8. CREATE a config/settings file for embedding preferences:
   - PRIMARY_EMBEDDING_PROVIDER: "local" | "cohere"
   - FALLBACK_ENABLED: true/false
   - LOCAL_MODEL_NAME: configurable
   - COHERE_MODEL_NAME: configurable

9. PERFORMANCE: Local embeddings should use GPU if available, CPU otherwise

10. DOCUMENTATION: Update README with:
    - Why we're not using OpenAI
    - Local embeddings benefits (privacy, cost, speed)
    - Cohere option for when you need it
    - How to switch between providers
    - Model dimension reference

ARCHITECTURE PRINCIPLE: 
Data sovereignty first. Local by default. API only when explicitly chosen.
No vendor lock-in. No IP thieves.

Files to modify:
- pyproject.toml (dependencies)
- .env.example (remove OpenAI, add optional Cohere)
- src/embeddings/ (new module with provider classes)
- src/processing/embedding_worker.py (use new providers)
- src/storage/weaviate_client.py (update schema dimensions)
- docker-compose.yml (remove OpenAI mentions)
- README.md (update docs)
- Add: src/config/embedding_settings.py

Test it works:
1. Should run with NO API keys (local only)
2. Should work with COHERE_API_KEY if provided
3. Should never call OpenAI
4. Should be faster than old implementation
5. Should maintain same API interface for rest of system