from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging

class CoHereProvider(LLMInterface):
    def __init__(self, api_key: str,
                       default_input_max_characters: int = 1000,
                       default_max_output_tokens: int = 1000,
                       default_temperature: float = 0.1):
        
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_max_output_tokens = default_max_output_tokens
        self.default_temperature = default_temperature
        
        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None
        
        self.client = cohere.ClientV2(
            api_key=self.api_key
        )
        
        self.logger = logging.getLogger(__name__)
        
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        
    def generate_text(self, prompt: str, chat_history: list = [], max_output_token: int = None, temperature: float = None):
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model was not set")
            return None
        
        max_output_token = max_output_token if max_output_token else self.default_max_output_tokens
        temperature = temperature if temperature else self.default_temperature
        
        chat_history.append(
            self.construct_prompt(prompt=prompt, role=CohereEnums.USER.value)
        )
        
        response = self.client.chat(
            model=self.generation_model_id,
            messages=chat_history,
            temperature=temperature,
            max_tokens=max_output_token
        )
        
        if not response or not response.message or not response.message.content or len(response.message.content) == 0 or not response.message.content[0].text:
            self.logger.error("Error while generating text")
            return None
        
        return response.message.content[0].text
    
    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model was not set")
            return None
        
        input_type = CoHereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.DOCUMENT.value:
            input_type = CoHereEnums.QUERY.value
            
        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(text)],
            input_type=input_type,
            embedding_types=['float'],
            output_dimension=self.embedding_size
        )
        
        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Error while embedding text")
            return None
        
        return response.embeddings.float[0]
        
    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(text=prompt)   
        }