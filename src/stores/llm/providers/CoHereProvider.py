from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
from typing import List, Union
import cohere
import logging


class CoHereProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        default_input_max_characters: int = 1000,
        default_max_output_tokens: int = 1000,
        default_temperature: float = 0.1
    ):

        self.api_key = api_key
        self.base_url = base_url

        self.default_input_max_characters = default_input_max_characters
        self.default_max_output_tokens = default_max_output_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None
        
        self.enums = CoHereEnums

        if self.base_url:
            self.client = cohere.ClientV2(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = cohere.ClientV2(
                api_key=self.api_key
            )

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_token: int = None,
        temperature: float = None
    ):

        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model was not set")
            return None

        if chat_history is None:
            chat_history = []

        max_output_token = (
            max_output_token
            if max_output_token is not None
            else self.default_max_output_tokens
        )

        temperature = (
            temperature
            if temperature is not None
            else self.default_temperature
        )

        messages = list(chat_history)

        messages.append(
            self.construct_prompt(
                prompt=prompt,
                role=CoHereEnums.USER.value
            )
        )

        response = self.client.chat(
            model=self.generation_model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_token
        )

        if (
            not response
            or not response.message
            or not response.message.content
            or len(response.message.content) == 0
            or not getattr(response.message.content[0], "text", None)
        ):
            self.logger.error("Error while generating text")
            return None

        return response.message.content[0].text

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None
        
        if isinstance(text, str):
            text = [text]
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was not set")
            return None
        
        input_type = CoHereEnums.DOCUMENT
        if document_type == DocumentTypeEnum.QUERY:
            input_type = CoHereEnums.QUERY

        response = self.client.embed(
            model = self.embedding_model_id,
            texts = [ self.process_text(t) for t in text ],
            input_type = input_type,
            embedding_types=['float'],
        )

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Error while embedding text with CoHere")
            return None
        
        return [ f for f in response.embeddings.float ]

        
    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def construct_prompt(self, prompt: str, role: str):

        return {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": "prompt"
                }
            ]
        }