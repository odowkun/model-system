from pydantic import BaseModel

class ContentRequest(BaseModel):
    user_skill: str

class CollaborativeRequest(BaseModel):
    user_id: int

class HybridRequest(BaseModel):
    user_id: int
    user_skill: str
