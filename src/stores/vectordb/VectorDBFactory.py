from .providers import QdrantDBProvider
from ...controllers import BaseController
from ...enums import VectorDBEnums
class VectorDBFactory:

    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider):
        if provider == VectorDBEnums.QDRANT:
            db_path = self.base_controller.get_database_path(self.config.VECTOR_DB_PATH)

            return QdrantDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )

            return None