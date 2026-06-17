from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory

settings = get_settings()
factory = LLMProviderFactory(settings)
f= factory.create(settings.GENERATION_BACKEND)
e= factory.create(settings.EMBEDDING_BACKEND)

f.set_generation_model(settings.GENERATION_MODEL_ID)
e.set_embedding_model(settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_SIZE)

response = f.generate_text("Hello, how are you?", chat_history=[{"role": "system", "content": "You are a helpful assistant."}])
print("Generated Text:", response)

embedding = e.embed_text("Hello, how are you?")
print("Embedding:", embedding)


