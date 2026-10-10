from fastapi import APIRouter

from services.nlp.modality_router import (
    NlpRoutingDecision,
    NlpRoutingRequest,
    route_nlp_request,
)


router = APIRouter(
    prefix="/nlp",
    tags=["Natural Language Processing"],
)


@router.post("/route", response_model=NlpRoutingDecision)
async def route_request(request: NlpRoutingRequest) -> NlpRoutingDecision:
    return route_nlp_request(request)