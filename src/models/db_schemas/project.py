from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId

class Project(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    project_id: str = Field(..., min_length=1) # project id here is the name of the folder where the data chunks are stored

    @validator('project_id')
    def validate_project_id(cls, value):
        if not value.isalnum():
            raise ValueError('project_id must be alphanumeric')
        
        return value

    # By default, Pydantic does not allow arbitrary types like ObjectId, so we need to enable that.
    class Config:
        arbitrary_types_allowed = True