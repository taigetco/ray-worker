import uuid
from typing import Dict, Optional, List, no_type_check
from typing_extensions import TypedDict
import logging
import torch
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
import numpy as np
from ray import serve

logger = logging.getLogger("ray.serve")

app = FastAPI()

class Document(TypedDict):
    text: str

class DocumentObj(TypedDict):
    index: int
    relevance_score: float
    document: Optional[Document]

class Rerank(TypedDict):
    id: str
    results: List[DocumentObj]

@serve.deployment
@serve.ingress(app)
class RerankModel:
    def __init__(
        self,
        model_path: str,
        use_fp16: bool = False,
        model_config: Optional[Dict] = None,
    ):
        self._model_path = model_path
        self._model_config = model_config or dict()
        self._use_fp16 = use_fp16
        self._rerank_type = self._auto_detect_type(model_path)
        self.load()
    
    @staticmethod
    def _auto_detect_type(model_path):
        """This method may not be stable due to the fact that the tokenizer name may be changed.
        Therefore, we only use this method for unknown model types."""
        from transformers import AutoTokenizer

        type_mapper = {
            "LlamaTokenizerFast": "LLM-based layerwise",
            "GemmaTokenizerFast": "LLM-based",
            "XLMRobertaTokenizerFast": "normal",
        }

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        rerank_type = type_mapper.get(type(tokenizer).__name__)
        if rerank_type is None:
            raise Exception(
                f"Can't determine the rerank type based on the tokenizer {tokenizer}"
            )
        return rerank_type
    
    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"model type, {self._rerank_type}")
        print(f"the device is {device}")
        if self._rerank_type == "normal":
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError:
                error_message = "Failed to import module 'sentence-transformers'"
                installation_guide = [
                    "Please make sure 'sentence-transformers' is installed. ",
                    "You can install it by `pip install sentence-transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = CrossEncoder(self._model_path, **self._model_config)
            if self._use_fp16:
                self._model.model.half()
        else:
            try:
                if self._rerank_type == "LLM-based":
                    from FlagEmbedding import FlagLLMReranker as FlagReranker
                elif self._rerank_type == "LLM-based layerwise":
                    from FlagEmbedding import LayerWiseFlagLLMReranker as FlagReranker
                else:
                    raise RuntimeError(
                        f"Unsupported Rank model type: {self._rerank_type}"
                    )
            except ImportError:
                error_message = "Failed to import module 'FlagEmbedding'"
                installation_guide = [
                    "Please make sure 'FlagEmbedding' is installed. ",
                    "You can install it by `pip install FlagEmbedding`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        return_documents: Optional[bool],
        **kwargs,
    ) -> Rerank:
        assert self._model is not None
        if kwargs:
            raise ValueError("rerank hasn't support extra parameter.")
        sentence_combinations = [[query, doc] for doc in documents]
        if self._rerank_type == "normal":
            similarity_scores = self._model.predict(sentence_combinations)
        else:
            similarity_scores = self._model.compute_score(sentence_combinations)
        sim_scores_argsort = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            sim_scores_argsort = sim_scores_argsort[:top_n]
        if return_documents:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=Document(text=documents[arg]),
                )
                for arg in sim_scores_argsort
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=None,
                )
                for arg in sim_scores_argsort
            ]
        return Rerank(id=str(uuid.uuid1()), results=docs)
    
 
    @app.post("/v1/rerank")
    async def rerank_api(self, request: Request):
        req = await request.json()
        query = req["query"]
        documents = req["documents"]
        top_n = req.get("top_n")
        return_documents = req.get("return_documents")
        print(f"query {query}, documents {documents}")

        return self.rerank(documents=documents, query=query, top_n=top_n, return_documents=return_documents)

def build_rerank_app(args: Dict[str, str]) -> serve.Application:
    
    model = args["model"]
    if model is None:
        print(f"not specified embedding model")
    
    return RerankModel.bind(model)