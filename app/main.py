from fastapi import FastAPI
from recommendation import content_based_filtering, collaborative_filtering, hybrid_recommendation
from dto import ContentRequest, CollaborativeRequest, HybridRequest

app = FastAPI()

@app.post("/recommend/content-based/")
def content_based(request: ContentRequest):
    return content_based_filtering(request)

@app.post("/recommend/collaborative/")
def collaborative(request: CollaborativeRequest):
    return collaborative_filtering(request)

@app.post("/recommend/hybrid/")
def hybrid(request: HybridRequest):
    return hybrid_recommendation(request)
