from pydantic import BaseModel
from typing import Optional

class ProcessRequestSchema(BaseModel):
    file_id: str
    chunk_size: Optional[int] = 100 # default chunk size
    overlap_size: Optional[int] = 20 # default overlap size
    do_reset: Optional[int] = 0 