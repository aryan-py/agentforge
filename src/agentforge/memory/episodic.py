"""Episodic memory — Qdrant-backed storage of past job execution experiences."""

import json
import logging
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExperienceRecord(BaseModel):
    """A record of a completed job execution stored for future learning.

    When the Trainer encounters a new job, it recalls similar ExperienceRecords
    and uses them to skip or accelerate web research.
    """

    experience_id: str = Field(default_factory=lambda: str(uuid4()))
    job_type: str
    domain: str
    tenant_id: str
    job_description_summary: str  # first 200 chars of description
    research_queries_used: List[str]
    skills_discovered: List[str]
    tool_types_needed: List[str]
    tools_that_worked: List[str]  # tool_ids actually called by worker
    quality_score: float
    research_confidence: float
    lessons: List[str]  # extracted by LLM post-evaluation
    approach_summary: str  # what approach worked
    duration_seconds: float
    cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EpisodicMemory:
    """Qdrant-backed memory that stores and retrieves past job execution experiences.

    Enables the TrainerAgent to:
    - Skip web research for high-quality past matches (quality >= 0.8)
    - Inject learned lessons into the expert_approach for medium matches
    - Improve tool selection based on what actually worked
    """

    COLLECTION_NAME = "agentforge_episodes"
    VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small

    def __init__(self, qdrant_url: str = "http://localhost:6333", openai_api_key: str = ""):
        self.qdrant_url = qdrant_url
        self.openai_api_key = openai_api_key
        self._client = None
        self._embedder = None

    async def initialize(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = AsyncQdrantClient(url=self.qdrant_url)
            self._embedder = self._make_embedder()

            collections = await self._client.get_collections()
            existing = [c.name for c in collections.collections]

            if self.COLLECTION_NAME not in existing:
                await self._client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")
            else:
                logger.info(f"Qdrant collection exists: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.warning(f"EpisodicMemory.initialize() failed (Qdrant may not be running): {e}")
            self._client = None

    def _make_embedder(self):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    async def _embed(self, text: str) -> List[float]:
        if self._embedder is None:
            self._embedder = self._make_embedder()
        return await self._embedder.aembed_query(text)

    async def remember(self, record: ExperienceRecord) -> None:
        """Store an experience record in Qdrant."""
        if self._client is None:
            logger.debug("EpisodicMemory not initialized — skipping remember()")
            return
        try:
            from qdrant_client.models import PointStruct

            embed_text = f"{record.job_type} {record.domain} {record.approach_summary}"
            vector = await self._embed(embed_text)
            payload = json.loads(record.model_dump_json())

            await self._client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[PointStruct(id=record.experience_id, vector=vector, payload=payload)],
            )
            logger.info(f"Stored experience {record.experience_id} for {record.job_type}/{record.domain}")
        except Exception as e:
            logger.warning(f"EpisodicMemory.remember() failed: {e}")

    async def recall(
        self,
        job,  # JobDefinition
        top_k: int = 3,
        min_quality: float = 0.65,
    ) -> List[ExperienceRecord]:
        """Retrieve past experiences similar to the given job."""
        if self._client is None:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

            query_text = f"{job.job_type} {job.domain} {job.description[:100]}"
            vector = await self._embed(query_text)

            must_conditions = [
                FieldCondition(key="tenant_id", match=MatchValue(value=job.tenant_id)),
                FieldCondition(key="quality_score", range=Range(gte=min_quality)),
            ]

            results = await self._client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=vector,
                limit=top_k,
                query_filter=Filter(must=must_conditions),
                with_payload=True,
            )

            records = []
            for hit in results:
                try:
                    records.append(ExperienceRecord(**hit.payload))
                except Exception as e:
                    logger.debug(f"Could not parse experience record: {e}")
            return records
        except Exception as e:
            logger.warning(f"EpisodicMemory.recall() failed: {e}")
            return []

    async def extract_lessons(
        self,
        job,  # JobDefinition
        worker_result,  # WorkerResult
        evaluation,  # EvaluationResult
        llm: BaseChatModel,
    ) -> List[str]:
        """Use LLM to extract 2-3 specific lessons from a completed job."""
        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content="You analyze completed AI agent jobs to extract lessons. "
                        "Write 2-3 specific lessons that would help an agent do better next time. "
                        "Be concrete. Focus on: which tools were most useful, what approach worked, what to avoid. "
                        "Return a JSON array of strings only. No markdown."
                    ),
                    HumanMessage(
                        content=f"Job: {job.description[:200]}\n"
                        f"Domain: {job.domain}\n"
                        f"Tools used: {worker_result.tools_called}\n"
                        f"Quality score: {evaluation.quality_score:.2f}\n"
                        f"Feedback: {evaluation.feedback}\n"
                        f"Criteria met: {evaluation.criteria_met}\n"
                        f"Criteria failed: {evaluation.criteria_failed}"
                    ),
                ]
            )
            raw = response.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            logger.warning(f"extract_lessons() failed: {e}")
            return [
                f"Tools {worker_result.tools_called} were used for {job.job_type}",
                f"Quality score was {evaluation.quality_score:.0%}",
            ]

    async def stats(self, tenant_id: str = "*") -> dict:
        """Return memory statistics."""
        if self._client is None:
            return {"total_experiences": 0, "avg_quality": 0.0, "most_used_tools": [], "domains_covered": []}
        try:
            info = await self._client.get_collection(self.COLLECTION_NAME)
            return {
                "total_experiences": info.points_count,
                "avg_quality": 0.0,
                "most_used_tools": [],
                "domains_covered": [],
            }
        except Exception as e:
            logger.warning(f"EpisodicMemory.stats() failed: {e}")
            return {"total_experiences": 0, "error": str(e)}
