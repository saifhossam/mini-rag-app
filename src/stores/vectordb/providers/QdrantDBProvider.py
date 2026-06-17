from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from qdrant_client import models, QdrantClient
import logging

class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_path: str, distance_method: str):
        self.client = None
        self.db_path = db_path
        self.distance_method = distance_method

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name)
    
    def list_all_collections(self):
        return self.client.get_collections().collections

    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name).dict()

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name)

    def create_collection(self, collection_name: str,
                          embedding_size: int,
                          do_reset: bool = False):
        if do_reset and self.is_collection_existed(collection_name):
            self.delete_collection(collection_name)

        if not self.is_collection_existed(collection_name):
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_size, distance=self.distance_method)
            )

            return True

        return False

    def insert_one(self, collection_name: str, text: str, vector: list,
                   metadata: dict = None,
                   record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            return False

        payload = {"text": text,
                   "metadata": metadata if metadata else {}}
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(id=record_id, vector=vector, payload=payload)
                ]
            )
        except Exception as e:
            self.logger.error(f"Error occurred while inserting point into collection {collection_name}: {e}")
            return False
        
        return True

    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadata: list = None,
                    record_ids: list = None, batch_size: int = 50):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            return False

        points = []
        for i in range(len(texts)):
            payload = {"text": texts[i],
                       "metadata": metadata[i] if metadata else {}}
            point_id = record_ids[i] if record_ids else None
            points.append(models.PointStruct(id=point_id, vector=vectors[i], payload=payload))

        try:
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
        except Exception as e:
            self.logger.error(f"Error occurred while inserting points into collection {collection_name}: {e}")
            return False
        
        return True

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 10):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            return []

        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit
            )
            return search_result
        except Exception as e:
            self.logger.error(f"Error occurred while searching in collection {collection_name}: {e}")
            return []