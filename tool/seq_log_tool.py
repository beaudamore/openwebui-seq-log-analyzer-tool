"""
title: Seq Log Analysis Tool
author: Beau D'Amore www.damore.ai
version: 1.0.0
description: Deep analysis of Seq logs with native OpenWebUI integrations.
requirements: fastapi, pandas, spacy, nltk
"""

import re
import json
import inspect
import requests
from datetime import datetime
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
import pandas as pd
import spacy
from fastapi import Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field

from open_webui.models.knowledge import Knowledges, KnowledgeUserModel
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
from open_webui.routers.retrieval import (
    ProcessFileForm,
    QueryCollectionsForm,
    process_file,
    query_collection_handler,
)

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class SeqToolError(Exception):
    """Base exception for Seq log tool errors."""


class EventEmitter:
    """Centralized emitter providing consistent phases and formatting."""

    def __init__(self, event_emitter=None):
        self.event_emitter = event_emitter

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        seq_server_url: str = Field(
            default="http://localhost:5341", # Placeholder, user should configure
            description="Seq server base URL (e.g., http://localhost:5341)",
        )
        seq_api_key: str = Field(
            default="",
            description="Seq API Key",
        )
        default_knowledge_base: str = Field(
            default="SEQ Logs",
            description="Default knowledge base name or ID to search/store data",
        )
        enable_hybrid_search: bool = Field(
            default=False,
            description="Enable hybrid search (combines semantic and keyword search)",
        )
        max_events: int = Field(
            default=50,
            description="Maximum number of events to retrieve per search",
        )
        reranker_results: int = Field(
            default=0,
            description="Number of results to retain after reranking (0 disables reranking)",
        )
        relevance_threshold: float = Field(
            default=0.0,
            description="Minimum relevance score threshold for results (0.0-1.0)",
        )
        enable_debug_output: bool = Field(
            default=True,
            description="Include debug information in responses",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def analyze_seq_logs(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        signal: Optional[str] = None,
        count: Optional[int] = None,
        render: Optional[bool] = None,
        __event_emitter__=None,
        __user__=None,
        __request__: Optional[Any] = None,
    ) -> str:
        """
        🔍 Analyze Seq Logs Tool for OpenWebUI

        Search and analyze Seq logs with full API capabilities. Performs intelligent analysis by:
        1. Querying existing knowledge container first (direct DB access)
        2. Fetching from Seq API if needed (using the query as a filter)
        3. Processing data with NLP (spaCy, NLTK)
        4. Embedding into OpenWebUI knowledge container
        5. Returning RAG-enhanced results

        IMPORTANT FOR LLM USE:
        - The `query` parameter accepts both natural language AND Seq filter syntax
        - Use `discover_seq_metadata()` first to see available properties, signals, and values
        - Log levels MUST be UPPERCASE: 'ERROR', 'WARNING', 'INFO' (not 'Error' or 'error')
        - Time ranges go in `from_date` and `to_date` parameters, NOT in the filter expression
        - Seq filter syntax uses ==, !=, >, <, >=, <=, &&, || operators and @ for built-in properties
        
        PARAMETERS:
        
        query (str, REQUIRED): 
            Natural language query OR Seq filter expression.
            Examples (Natural Language):
            - "Show me all errors in production from the last 24 hours"
            - "Find database connection failures in QA environment"
            - "Analyze slow requests (>1000ms) in CD role"
            - "Show me warnings in STAGE for CM role"
            
            Examples (Seq Filter Syntax):
            - "@Level == 'ERROR' && ENV == 'PROD'"
            - "ENV == 'QA' && ROLE == 'CM'"
            - "@Level == 'WARNING' && ENV == 'STAGE' && ROLE == 'CD'"
            - "@Level == 'INFO'" (use uppercase: ERROR, WARNING, INFO)
            - "'database connection failed'" (text search requires single quotes)
            - "Duration > 1000 && ENV == 'PROD'"
            - "ROLE == 'CD' && @Level == 'ERROR'"
        
        from_date (str, optional): 
            ISO 8601 start date for time range filtering (e.g., "2023-01-01T00:00:00Z", "2024-12-18T00:00:00Z").
            Use this for date ranges, NOT in the filter expression.
            Examples: "2024-12-17T00:00:00Z", "2024-12-18T12:00:00Z"
        
        to_date (str, optional): 
            ISO 8601 end date for time range filtering (e.g., "2023-12-31T23:59:59Z").
            Examples: "2024-12-18T23:59:59Z", "2024-12-19T00:00:00Z"
        
        signal (str, optional): 
            Signal Name (e.g., "Production", "Errors", "UAT") or signal ID (e.g., "signal-123").
            Signals are predefined queries/filters in Seq. Use `discover_seq_metadata()` to see available signals.
            Examples: "Production", "Errors", "UAT", "signal-abc123"
        
        count (int, optional): 
            Maximum number of events to retrieve. Overrides the `max_events` valve setting.
            Default uses valve setting (typically 50).
            Examples: 100, 200, 500
        
        render (bool, optional): 
            Whether to render message templates with property values. 
            Default: true (messages are rendered with values substituted).
            Set to false for raw message templates.

        AVAILABLE SEQ PROPERTIES (from discover_seq_metadata):
        Standard Properties:
        - @Level: Log level (ERROR, WARNING, INFO, DEBUG, VERBOSE, FATAL)
        - @Message: Log message text
        - @MessageTemplate: Message template
        - @Exception: Exception details
        - @Timestamp: Event timestamp
        - @Properties: All event properties
        - @EventType: Event type identifier
        
        Custom Properties (examples from your Seq instance):
        - ENV: Environment (PROD, QA, STAGE)
        - ROLE: Role (CD, CM)
        - Duration: Request duration in milliseconds
        - Application: Application name
        - SLOT: Deployment slot
        - StatusCode: HTTP status code
        
        Use `discover_seq_metadata()` to see the complete list of properties and values in your Seq instance.

        FILTER SYNTAX RULES:
        - Equality: @Level == 'ERROR', ENV == 'PROD'
        - Inequality: @Level != 'INFO', Duration != 0
        - Comparison: Duration > 1000, StatusCode >= 400
        - Logical AND: @Level == 'ERROR' && ENV == 'PROD'
        - Logical OR: @Level == 'ERROR' || @Level == 'WARNING'
        - Text search: 'database connection' (wrap in single quotes)
        - Property references: @Level, @Message, ENV, ROLE, Duration
        - IMPORTANT: Log levels are CASE-SENSITIVE and must be UPPERCASE

        COMMON QUERY PATTERNS:
        - Errors in production: "@Level == 'ERROR' && ENV == 'PROD'"
        - Slow requests: "Duration > 1000"
        - Specific environment and role: "ENV == 'QA' && ROLE == 'CM'"
        - Multiple levels: "@Level == 'ERROR' || @Level == 'WARNING'"
        - Text search with filters: "'timeout' && ENV == 'PROD'"
        - Recent errors: "@Level == 'ERROR'" with from_date="2024-12-18T00:00:00Z"

        OUTPUT:
        Returns a comprehensive report including:
        - Existing knowledge from previous searches (RAG retrieval)
        - New events with timestamp, level, message, exception details
        - Named entities extracted via NLP
        - Full event details (properties, stack traces)
        - Knowledge base update status
        - Debug information (if enabled in valves)
        """

        import logging
        logging.error(f"🚀 === ANALYZE_SEQ_LOGS ENTRY ===")
        logging.error(f"   query: '{query}'")
        logging.error(f"   from_date: {from_date}")
        logging.error(f"   to_date: {to_date}")
        logging.error(f"   signal: {signal}")
        logging.error(f"   count: {count}")
        logging.error(f"   render: {render}")
        
        async def _noop_event(*_args, **_kwargs):
            return None

        logging.error(f"🔵 Creating eventer...")
        eventer = __event_emitter__ if __event_emitter__ is not None else _noop_event
        logging.error(f"   eventer type: {type(eventer)}")

        logging.error(f"🔵 Creating EventEmitter...")
        emitter = EventEmitter(eventer)
        logging.error(f"   emitter created: {emitter}")
        
        logging.error(f"🔵 Calling progress_update...")
        await emitter.progress_update("🔍 Initializing Seq log analysis...")
        logging.error(f"   progress_update complete")

        logging.error(f"🔵 Reading valves.max_events...")
        effective_max_results = self.valves.max_events
        logging.error(f"   effective_max_results: {effective_max_results}")
        
        logging.error(f"🔵 Building settings_info string...")
        settings_info = f"Max Events: {effective_max_results}"
        logging.error(f"   settings_info (initial): '{settings_info}'")
        
        logging.error(f"🔵 Checking valves.reranker_results...")
        if self.valves.reranker_results > 0:
            logging.error(f"   reranker_results > 0: {self.valves.reranker_results}")
            settings_info += f", Reranker: {self.valves.reranker_results}"
            logging.error(f"   settings_info (updated): '{settings_info}'")
        else:
            logging.error(f"   reranker_results <= 0: {self.valves.reranker_results}")
            
        logging.error(f"🔵 Calling progress_update with settings...")
        await emitter.progress_update(f"⚙️ Settings: {settings_info}")
        logging.error(f"   progress_update complete")

        try:
            logging.error(f"🔵 Calling require_request...")
            request = KnowledgeRepository.require_request(__request__)
            logging.error(f"   request: {request}")
            
            logging.error(f"🔵 Calling resolve_user...")
            user = await KnowledgeRepository.resolve_user(__user__)
            logging.error(f"   user.id: {user.id if user else 'None'}")
        except ValueError as exc:
            logging.error(f"❌ ValueError caught: {exc}")
            msg = f"❌ {exc}"
            logging.error(f"🔵 Calling error_update...")
            await emitter.error_update(msg)
            logging.error(f"🔵 Returning error message")
            return msg

        logging.error(f"🔵 Checking valves.default_knowledge_base...")
        logging.error(f"   default_knowledge_base: '{self.valves.default_knowledge_base}'")
        if not self.valves.default_knowledge_base:
            logging.error(f"   ❌ default_knowledge_base is empty")
            msg = "❌ No default knowledge base configured. Set one in the tool configuration."
            await emitter.error_update(msg)
            return msg
        else:
            logging.error(f"   ✅ default_knowledge_base is set")

        logging.error(f"🔵 Initializing seq_debug variable...")
        seq_debug = ""
        logging.error(f"   seq_debug (initial): '{seq_debug}'")
        
        logging.error(f"🔵 Checking valves.enable_debug_output...")
        logging.error(f"   enable_debug_output: {self.valves.enable_debug_output}")
        if self.valves.enable_debug_output:
            logging.error(f"   ✅ Building seq_debug string...")
            seq_debug = f"""🔧 **Debug Information**:
- Query/Filter: '{query}'
- Knowledge Base: '{self.valves.default_knowledge_base}'
- Max Events: {self.valves.max_events}
- Server: {self.valves.seq_server_url}
"""
            logging.error(f"   seq_debug: {seq_debug}")
        else:
            logging.error(f"   ❌ Debug output disabled")

        try:
            # 1. Check Knowledge Base
            logging.error(f"🔍 === KNOWLEDGE BASE LOOKUP START ===")
            logging.error(f"   user.id: {user.id}")
            logging.error(f"   kb_name: '{self.valves.default_knowledge_base}'")
            logging.error(f"   permission: 'write'")
            
            logging.error(f"🔵 Calling find_by_name...")
            kb = await KnowledgeRepository.find_by_name(
                user.id,
                self.valves.default_knowledge_base,
                permission="write",
            )
            logging.error(f"   kb returned: {kb}")
            logging.error(f"   kb type: {type(kb)}")
            if kb:
                logging.error(f"   kb.id: {kb.id}")
                logging.error(f"   kb.name: {kb.name}")
            
            logging.error(f"🔵 Checking if kb exists...")
            if not kb:
                # Try to create it if it doesn't exist? 
                # For now, just error like the original tool, or maybe we should just proceed to create it implicitly via upload?
                # The original tool errors if not found.
                logging.error(f"   ❌ KB not found!")
                logging.error(f"🔵 Loading available KBs...")
                available = await KnowledgeRepository.load_by_user(user.id, "write")
                logging.error(f"   available count: {len(available) if available else 0}")
                
                logging.error(f"🔵 Building names list...")
                names = [item.name or item.id for item in available]
                logging.error(f"   names: {names}")
                
                logging.error(f"🔵 Building error_msg...")
                error_msg = (
                    f"Knowledge container '{self.valves.default_knowledge_base}' not found."
                    f" Available knowledge bases: {', '.join(names) if names else 'none'}"
                )
                logging.error(f"   error_msg: '{error_msg}'")
                
                logging.error(f"🔵 Calling error_update...")
                await emitter.error_update(error_msg)
                logging.error(f"🔵 Returning error...")
                return seq_debug + f"❌ {error_msg}"

            logging.error(f"   ✅ KB found!")
            logging.error(f"🔵 Calling progress_update...")
            await emitter.progress_update(f"🔍 Querying knowledge container: {kb.name or kb.id}")
            logging.error(f"   progress_update complete")

            # Create a better semantic query from the Seq filter for RAG search
            # Extract key terms from filter to search stored logs
            logging.error(f"🔍 === SEMANTIC QUERY BUILDING START ===")
            logging.error(f"   query: '{query}'")
            
            logging.error(f"🔵 Initializing semantic_query...")
            semantic_query = query
            logging.error(f"   semantic_query (initial): '{semantic_query}'")
            
            logging.error(f"🔵 Checking if query exists...")
            if query:
                logging.error(f"   ✅ query exists, extracting components...")
                
                # Extract level (ERROR, WARNING, etc.)
                logging.error(f"🔵 Searching for @Level...")
                level_match = re.search(r"@Level\s*==\s*['\"](\w+)['\"]", query, re.IGNORECASE)
                logging.error(f"   level_match: {level_match}")
                if level_match:
                    logging.error(f"   level_match.group(1): '{level_match.group(1)}'")
                    
                # Extract environment (PROD, QA, STAGE)
                logging.error(f"🔵 Searching for ENV...")
                env_match = re.search(r"ENV\s*==\s*['\"](\w+)['\"]", query, re.IGNORECASE)
                logging.error(f"   env_match: {env_match}")
                if env_match:
                    logging.error(f"   env_match.group(1): '{env_match.group(1)}'")
                    
                # Extract role (CD, CM)
                logging.error(f"🔵 Searching for ROLE...")
                role_match = re.search(r"ROLE\s*==\s*['\"](\w+)['\"]", query, re.IGNORECASE)
                logging.error(f"   role_match: {role_match}")
                if role_match:
                    logging.error(f"   role_match.group(1): '{role_match.group(1)}'")
                
                # Build a natural language query for semantic search
                logging.error(f"🔵 Building query_parts list...")
                query_parts = []
                logging.error(f"   query_parts (initial): {query_parts}")
                
                logging.error(f"🔵 Checking level_match...")
                if level_match:
                    logging.error(f"   ✅ Adding level to query_parts")
                    query_parts.append(f"{level_match.group(1)} level")
                    logging.error(f"   query_parts: {query_parts}")
                    
                logging.error(f"🔵 Checking env_match...")
                if env_match:
                    logging.error(f"   ✅ Adding env to query_parts")
                    query_parts.append(f"{env_match.group(1)} environment")
                    logging.error(f"   query_parts: {query_parts}")
                    
                logging.error(f"🔵 Checking role_match...")
                if role_match:
                    logging.error(f"   ✅ Adding role to query_parts")
                    query_parts.append(f"{role_match.group(1)} role")
                    logging.error(f"   query_parts: {query_parts}")
                
                logging.error(f"🔵 Checking if query_parts has content...")
                if query_parts:
                    logging.error(f"   ✅ query_parts has content, building semantic_query")
                    semantic_query = " ".join(query_parts) + " logs events"
                    logging.error(f"   semantic_query: '{semantic_query}'")
                else:
                    logging.error(f"   ❌ query_parts is empty, using fallback")
                    # Fallback: just search for "error logs" or similar generic terms
                    semantic_query = "error exception logs events"
                    logging.error(f"   semantic_query (fallback): '{semantic_query}'")
            else:
                logging.error(f"   ❌ query is empty")
            
            logging.error(f"🔍 === SEMANTIC QUERY BUILDING END ===")
            logging.error(f"   Final semantic_query: '{semantic_query}'")
            
            logging.error(f"🔵 Calling progress_update with semantic query...")
            await emitter.progress_update(f"🔎 Semantic search query: '{semantic_query}'")
            logging.error(f"   progress_update complete")
            
            # Try to query knowledge base - may fail if collection doesn't exist yet
            existing_text = ""
            existing_ids: set[str] = set()
            has_existing_data = False
            
            import logging
            logging.error(f"🔍 KB QUERY DEBUG START:")
            logging.error(f"   kb.id: {kb.id}")
            logging.error(f"   kb.name: {kb.name}")
            logging.error(f"   semantic_query: '{semantic_query}'")
            logging.error(f"   limit: {self.valves.max_events}")
            
            try:
                logging.error(f"   🔵 Calling query_knowledge_base...")
                rag_result = await KnowledgeRepository.query_knowledge_base(
                    request=request,
                    user=user,
                    kb_id=kb.id,
                    query=semantic_query,
                    limit=self.valves.max_events,
                    valves=self.valves,
                )
                logging.error(f"   🔵 RAG result type: {type(rag_result)}")
                logging.error(f"   🔵 RAG result keys: {rag_result.keys() if rag_result else 'None'}")
                
                documents = rag_result.get("documents", [])
                logging.error(f"   🔵 documents type: {type(documents)}")
                logging.error(f"   🔵 documents length: {len(documents) if documents else 0}")
                logging.error(f"   🔵 documents value: {documents}")
                
                if documents and len(documents) > 0 and documents[0]:
                    logging.error(f"   🔵 documents[0] type: {type(documents[0])}")
                    logging.error(f"   🔵 documents[0] length: {len(documents[0]) if documents[0] else 0}")
                    logging.error(f"   🔵 First 500 chars of documents[0]: {str(documents[0])[:500]}")
                    
                    doc_list = documents[0]
                    existing_text = "\n\n".join(doc_list)
                    logging.error(f"   🔵 existing_text length: {len(existing_text)}")
                    logging.error(f"   🔵 First 200 chars of existing_text: {existing_text[:200]}")
                    
                    # Extract Seq Event IDs if present in the text
                    existing_ids = set(re.findall(r"Event ID:\s*([A-Za-z0-9_-]+)", existing_text))
                    logging.error(f"   🔵 existing_ids found: {existing_ids}")
                    logging.error(f"   🔵 existing_ids count: {len(existing_ids)}")
                    
                    if existing_ids:
                        has_existing_data = True
                        logging.error(f"   ✅ has_existing_data = True")
                        await emitter.progress_update(f"📚 Found {len(existing_ids)} existing events in knowledge base")
                        
                        # IMPORTANT: Never use cache alone for time-range queries
                        # Cache may not contain full requested time range
                        # Only use cache as supplementary data, always fetch from API to ensure completeness
                        logging.error(f"   ℹ️ Cache found but will still fetch from API to ensure complete results")
                        await emitter.progress_update("ℹ️ Found cached data, but fetching from API to ensure complete time range coverage")
                    else:
                        logging.error(f"   ⚠️ No Event IDs extracted from existing_text")
                else:
                    logging.error(f"   ⚠️ No documents or documents[0] is empty")
            except Exception as kb_query_error:
                # Knowledge base may not exist yet or collection not initialized
                logging.error(f"   ❌ KB query exception: {kb_query_error}")
                logging.error(f"   ❌ Exception type: {type(kb_query_error)}")
                import traceback
                logging.error(f"   ❌ Traceback: {traceback.format_exc()}")
                await emitter.progress_update(f"ℹ️ Knowledge base query failed (may be empty): {str(kb_query_error)[:100]}")
                has_existing_data = False
            
            logging.error(f"   🔵 Final has_existing_data: {has_existing_data}")
            logging.error(f"🔍 KB QUERY DEBUG END")
            
            logging.error(f"🔵 Checking has_existing_data...")
            if not has_existing_data:
                logging.error(f"   ❌ has_existing_data is False")
                logging.error(f"🔵 Calling progress_update...")
                await emitter.progress_update("ℹ️ No existing knowledge found, will fetch from Seq API")
                logging.error(f"   progress_update complete")
            else:
                logging.error(f"   ✅ has_existing_data is True")

            # 2. Fetch from Seq API (only if we don't have sufficient cached data)
            logging.error(f"🌐 === SEQ API FETCH START ===")
            logging.error(f"🔵 Calling progress_update...")
            await emitter.progress_update("🌐 Connecting to Seq API...")
            logging.error(f"   progress_update complete")
            
            # Clean up query - remove invalid time expressions that LLM might add to the filter
            # (time should be in fromDate/toDate parameters, NOT in the filter)
            logging.error(f"🔵 Cleaning up query...")
            logging.error(f"   query (before cleanup): '{query}'")
            if query:
                logging.error(f"   ✅ query exists, applying regex cleanup...")
                
                logging.error(f"🔵 Removing TimeGenerated expressions...")
                query = re.sub(r"\s*&&\s*['\"]?@?TimeGenerated[^&]+", "", query, flags=re.IGNORECASE)
                logging.error(f"   query after TimeGenerated: '{query}'")
                
                logging.error(f"🔵 Removing Timestamp expressions...")
                query = re.sub(r"\s*&&\s*['\"]?@?Timestamp[^&]+", "", query, flags=re.IGNORECASE)
                logging.error(f"   query after Timestamp: '{query}'")
                
                logging.error(f"🔵 Removing relative time expressions...")
                query = re.sub(r"\s*&&\s*['\"]?(yesterday|today|last \d+|past \d+)['\"]?", "", query, flags=re.IGNORECASE)
                logging.error(f"   query after relative times: '{query}'")
                
                logging.error(f"🔵 Stripping whitespace and &&...")
                query = query.strip().strip("&&").strip()
                logging.error(f"   query (final): '{query}'")
            else:
                logging.error(f"   ❌ query is empty, skipping cleanup")
            
            # Resolve signal if provided
            logging.error(f"🔵 Initializing resolved_signal_id...")
            resolved_signal_id = None
            logging.error(f"   resolved_signal_id: {resolved_signal_id}")
            
            logging.error(f"🔵 Checking if signal provided...")
            logging.error(f"   signal: {signal}")
            if signal:
                logging.error(f"   ✅ signal provided: '{signal}'")
                logging.error(f"🔵 Calling progress_update...")
                await emitter.progress_update(f"🔍 Resolving signal '{signal}'...")
                logging.error(f"   progress_update complete")
                
                try:
                    logging.error(f"🔵 Building signal URL...")
                    sig_url = f"{self.valves.seq_server_url.rstrip('/')}/api/signals"
                    logging.error(f"   sig_url: '{sig_url}'")
                    
                    logging.error(f"🔵 Building signal headers...")
                    sig_headers = {"X-Seq-ApiKey": self.valves.seq_api_key}
                    logging.error(f"   sig_headers: {sig_headers}")
                    
                    logging.error(f"🔵 Calling requests.get for signals...")
                    sig_resp = await run_in_threadpool(requests.get, sig_url, headers=sig_headers, timeout=5)
                    logging.error(f"   sig_resp.status_code: {sig_resp.status_code}")
                    
                    logging.error(f"🔵 Checking sig_resp.status_code...")
                    if sig_resp.status_code == 200:
                        logging.error(f"   ✅ status_code is 200")
                        
                        logging.error(f"🔵 Parsing JSON response...")
                        signals_list = sig_resp.json()
                        logging.error(f"   signals_list length: {len(signals_list) if signals_list else 0}")
                        
                        # Check for Title match first (case-insensitive)
                        logging.error(f"🔵 Searching for Title match (case-insensitive)...")
                        for s in signals_list:
                            s_title = s.get("Title", "")
                            logging.error(f"   Checking: '{s_title}' vs '{signal}'")
                            if s_title.lower() == signal.lower():
                                resolved_signal_id = s.get("Id")
                                logging.error(f"   ✅ MATCH FOUND! resolved_signal_id: {resolved_signal_id}")
                                break
                        
                        # If not found, check if it might be an ID
                        logging.error(f"🔵 Checking if resolved_signal_id found...")
                        if not resolved_signal_id:
                            logging.error(f"   ❌ No Title match, checking for ID match...")
                            for s in signals_list:
                                s_id = s.get("Id")
                                logging.error(f"   Checking ID: '{s_id}' vs '{signal}'")
                                if s_id == signal:
                                    resolved_signal_id = s.get("Id")
                                    logging.error(f"   ✅ ID MATCH FOUND! resolved_signal_id: {resolved_signal_id}")
                                    break
                        
                        logging.error(f"🔵 Final check on resolved_signal_id...")
                        if resolved_signal_id:
                            logging.error(f"   ✅ resolved_signal_id found: {resolved_signal_id}")
                            logging.error(f"🔵 Calling progress_update...")
                            await emitter.progress_update(f"✅ Found signal ID: {resolved_signal_id}")
                            logging.error(f"   progress_update complete")
                        else:
                            logging.error(f"   ❌ resolved_signal_id NOT found")
                            logging.error(f"🔵 Calling progress_update...")
                            await emitter.progress_update(f"⚠️ Signal '{signal}' not found, ignoring.")
                            logging.error(f"   progress_update complete")
                    else:
                        logging.error(f"   ❌ status_code is not 200: {sig_resp.status_code}")
                        logging.error(f"🔵 Calling error_update...")
                        await emitter.error_update(f"⚠️ Failed to list signals: {sig_resp.status_code}")
                        logging.error(f"   error_update complete")
                except Exception as e:
                    logging.error(f"   ❌ Exception caught: {e}")
                    logging.error(f"   Exception type: {type(e)}")
                    import traceback
                    logging.error(f"   Traceback: {traceback.format_exc()}")
                    logging.error(f"🔵 Calling error_update...")
                    await emitter.error_update(f"⚠️ Error resolving signal: {e}")
                    logging.error(f"   error_update complete")
            else:
                logging.error(f"   ❌ signal not provided")

            logging.error(f"🔵 Defining seq_search function...")
            def seq_search(filter_expression: str, limit: int, start: Optional[str], end: Optional[str], signal_id: Optional[str] = None, use_render: Optional[bool] = None) -> List[Dict[str, Any]]:
                import logging
                logging.error(f"🔍 === SEQ_SEARCH FUNCTION ENTRY ===")
                logging.error(f"   filter_expression: '{filter_expression}'")
                logging.error(f"   limit: {limit}")
                logging.error(f"   start: {start}")
                logging.error(f"   end: {end}")
                logging.error(f"   signal_id: {signal_id}")
                logging.error(f"   use_render: {use_render}")
                
                logging.error(f"🔵 Building headers...")
                headers = {
                    "X-Seq-ApiKey": self.valves.seq_api_key,
                    "Content-Type": "application/json"
                }
                logging.error(f"   headers: {headers}")
                
                logging.error(f"🔵 Building params...")
                params = {
                    "count": limit,
                    "render": "true" if use_render is None else str(use_render).lower()
                }
                logging.error(f"   params (initial): {params}")
                
                # Strip outer quotes if LLM wrapped the filter in quotes
                logging.error(f"🔵 Checking filter_expression for outer quotes...")
                if filter_expression and filter_expression.strip():
                    logging.error(f"   ✅ filter_expression exists")
                    logging.error(f"🔵 Stripping whitespace...")
                    filter_expression = filter_expression.strip()
                    logging.error(f"   filter_expression (stripped): '{filter_expression}'")
                    
                    logging.error(f"🔵 Checking for double quotes...")
                    if filter_expression.startswith('"') and filter_expression.endswith('"'):
                        logging.error(f"   ✅ Has outer double quotes, removing...")
                        filter_expression = filter_expression[1:-1]
                        logging.error(f"   filter_expression (unquoted): '{filter_expression}'")
                    else:
                        logging.error(f"   ❌ No outer double quotes")
                else:
                    logging.error(f"   ❌ filter_expression is empty")
                
                # Only add filter if it looks like valid Seq syntax (contains ==, !=, >, <, etc.)
                # or is a text search (wrapped in single quotes)
                # Skip if it's likely natural language
                logging.error(f"🔵 Checking if filter_expression should be added to params...")
                if filter_expression and filter_expression.strip():
                    logging.error(f"   ✅ filter_expression exists")
                    
                    # Check if it looks like Seq filter syntax
                    logging.error(f"🔵 Defining seq_operators...")
                    seq_operators = ['==', '!=', '>=', '<=', '>', '<', '&&', '||', '@']
                    logging.error(f"   seq_operators: {seq_operators}")
                    
                    logging.error(f"🔵 Checking is_text_search...")
                    # Detect text-search syntax wrapped in single quotes
                    is_text_search = filter_expression.strip().startswith("'") and filter_expression.strip().endswith("'")
                    logging.error(f"   is_text_search: {is_text_search}")
                    
                    logging.error(f"🔵 Checking has_seq_syntax...")
                    has_seq_syntax = any(op in filter_expression for op in seq_operators)
                    logging.error(f"   has_seq_syntax: {has_seq_syntax}")
                    
                    logging.error(f"🔵 Checking if should add filter param...")
                    if has_seq_syntax or is_text_search:
                        logging.error(f"   ✅ Adding filter to params")
                        params["filter"] = filter_expression
                        logging.error(f"   params['filter']: '{params['filter']}'")
                    else:
                        logging.error(f"   ❌ Skipping filter (looks like natural language)")
                    # If it looks like natural language, skip the filter and rely on signal/date filtering
                else:
                    logging.error(f"   ❌ filter_expression is empty")
                    
                logging.error(f"🔵 Adding start date if provided...")
                if start:
                    logging.error(f"   ✅ start provided: {start}")
                    params["fromDate"] = start
                    logging.error(f"   params['fromDate']: {params['fromDate']}")
                else:
                    logging.error(f"   ❌ start not provided")
                    
                logging.error(f"🔵 Adding end date if provided...")
                if end:
                    logging.error(f"   ✅ end provided: {end}")
                    params["toDate"] = end
                    logging.error(f"   params['toDate']: {params['toDate']}")
                else:
                    logging.error(f"   ❌ end not provided")
                    
                logging.error(f"🔵 Adding signal_id if provided...")
                if signal_id:
                    logging.error(f"   ✅ signal_id provided: {signal_id}")
                    params["signal"] = signal_id
                    logging.error(f"   params['signal']: {params['signal']}")
                else:
                    logging.error(f"   ❌ signal_id not provided")

                logging.error(f"🔵 Building URL...")
                url = f"{self.valves.seq_server_url.rstrip('/')}/api/events"
                logging.error(f"   URL: {url}")
                
                # LOG THE QUERY BEING SENT
                logging.error(f"🔍 SEQ API CALL DEBUG:")
                logging.error(f"   URL: {url}")
                logging.error(f"   filter_expression input: '{filter_expression}'")
                logging.error(f"   Params being sent: {json.dumps(params, indent=2)}")
                logging.error(f"   from_date: {start}")
                logging.error(f"   to_date: {end}")
                logging.error(f"   signal_id: {signal_id}")
                
                logging.error(f"🔵 Calling requests.get...")
                response = requests.get(url, headers=headers, params=params, timeout=10)
                logging.error(f"   response.status_code: {response.status_code}")
                logging.error(f"   response length: {len(response.text) if response.text else 0}")
                
                logging.error(f"🔵 Checking response.status_code...")
                if response.status_code != 200:
                    logging.error(f"   ❌ status_code is not 200")
                    error_detail = response.text
                    logging.error(f"   error_detail: {error_detail}")
                    
                    logging.error(f"🔵 Checking if status_code is 400...")
                    if response.status_code == 400:
                        logging.error(f"   ✅ status_code is 400, adding hint")
                        error_detail += " (Hint: Query may need translation to Seq syntax. Try using discover_seq_metadata() to see available properties, or use Seq filter syntax like: @Level == 'Error' && ENV == 'Production')"
                        logging.error(f"   error_detail (with hint): {error_detail}")
                    
                    logging.error(f"🔵 Raising ValueError...")
                    raise ValueError(f"Seq API failed: {response.status_code} {error_detail}")
                
                logging.error(f"   ✅ status_code is 200")
                logging.error(f"🔵 Parsing JSON response...")
                events = response.json()
                logging.error(f"   events count: {len(events) if events else 0}")
                
                logging.error(f"🔵 Building results list...")
                results = []
                logging.error(f"   results (initial): {results}")
                
                logging.error(f"🔵 Iterating over events...")
                for idx, event in enumerate(events):
                    logging.error(f"   Processing event {idx + 1}/{len(events)}")
                    logging.error(f"   event.Id: {event.get('Id')}")
                    logging.error(f"   event.Level: {event.get('Level')}")
                    
                    result_item = {
                        "id": event.get("Id"),
                        "timestamp": event.get("Timestamp"),
                        "level": event.get("Level"),
                        "message": event.get("RenderedMessage", event.get("MessageTemplate")),
                        "exception": event.get("Exception"),
                        "properties": event.get("Properties", {})
                    }
                    logging.error(f"   result_item: {result_item}")
                    
                    results.append(result_item)
                    logging.error(f"   results count: {len(results)}")
                
                logging.error(f"🔵 Returning results...")
                logging.error(f"   Final results count: {len(results)}")
                return results

            # Override max results if count parameter provided
            logging.error(f"🔵 Checking count parameter...")
            logging.error(f"   count: {count}")
            logging.error(f"   effective_max_results: {effective_max_results}")
            actual_count = count if count is not None else effective_max_results
            logging.error(f"   actual_count: {actual_count}")
            
            logging.error(f"🔵 Calling seq_search via run_in_threadpool...")
            events = await run_in_threadpool(seq_search, query, actual_count, from_date, to_date, resolved_signal_id, render)
            logging.error(f"   events returned count: {len(events) if events else 0}")
            
            logging.error(f"🔵 Checking if events is empty...")
            if not events:
                logging.error(f"   ❌ events is empty")
                logging.error(f"🔵 Calling progress_update...")
                await emitter.progress_update("ℹ️ No events found in Seq for this filter.")
                logging.error(f"   progress_update complete")
                
                # If we have RAG results, we might still want to show them.
                logging.error(f"🔵 Checking if existing_text exists...")
                logging.error(f"   existing_text length: {len(existing_text) if existing_text else 0}")
                if not existing_text:
                    logging.error(f"   ❌ existing_text is empty, returning no_results")
                    logging.error(f"🔵 Building no_results message...")
                    no_results = f"❌ **Seq Analysis**: No events found for filter '{query}'"
                    logging.error(f"   no_results: '{no_results}'")
                    
                    logging.error(f"🔵 Calling eventer with result...")
                    await eventer({
                        "type": "result",
                        "data": {"description": no_results, "done": True, "hidden": False}
                    })
                    logging.error(f"   eventer complete")
                    
                    logging.error(f"🔵 Returning no_results...")
                    return seq_debug + no_results
                else:
                    logging.error(f"   ✅ existing_text has content, continuing...")

            logging.error(f"   ✅ events has {len(events)} items")
            logging.error(f"🔵 Calling progress_update...")
            await emitter.progress_update(f"📊 Processing {len(events)} events with NLP...")
            logging.error(f"   progress_update complete")

            # 3. Process Data
            logging.error(f"🔵 Defining process_data function...")
            def process_data(events_input: List[Dict[str, Any]]) -> pd.DataFrame:
                import logging
                logging.error(f"🔍 === PROCESS_DATA FUNCTION ENTRY ===")
                logging.error(f"   events_input count: {len(events_input) if events_input else 0}")
                
                logging.error(f"🔵 Checking if events_input is empty...")
                if not events_input:
                    logging.error(f"   ❌ events_input is empty, returning empty DataFrame")
                    return pd.DataFrame()
                
                logging.error(f"   ✅ events_input has data")
                logging.error(f"🔵 Creating DataFrame from events_input...")
                df = pd.DataFrame(events_input)
                logging.error(f"   df shape: {df.shape}")
                logging.error(f"   df columns: {list(df.columns)}")

                logging.error(f"🔵 Defining clean_text function...")
                def clean_text(text):
                    if not text:
                        return ""
                    text = re.sub(r"\s+", " ", str(text))
                    return text.lower()

                logging.error(f"🔵 Applying clean_text to df['message']...")
                df["clean_message"] = df["message"].apply(clean_text)
                logging.error(f"   df['clean_message'] created, first value: {df['clean_message'].iloc[0] if len(df) > 0 else 'N/A'}")
                
                logging.error(f"🔵 Loading stopwords...")
                stop_words = set(stopwords.words("english"))
                logging.error(f"   stop_words count: {len(stop_words)}")

                logging.error(f"🔵 Defining spacy_process function...")
                def spacy_process(text):
                    doc = nlp(text)
                    lemmas = [
                        token.lemma_
                        for token in doc
                        if not token.is_stop and token.is_alpha
                    ]
                    entities = [ent.text for ent in doc.ents]
                    entities = KnowledgeRepository.dedupe_preserve_order(entities)
                    return " ".join(lemmas), entities

                logging.error(f"🔵 Applying spacy_process to df['clean_message']...")
                df["lemmas_message"], df["entities_message"] = zip(
                    *df["clean_message"].apply(spacy_process)
                )
                logging.error(f"   df['lemmas_message'] created")
                logging.error(f"   df['entities_message'] created")
                if len(df) > 0:
                    logging.error(f"   First lemmas_message: {df['lemmas_message'].iloc[0][:100] if df['lemmas_message'].iloc[0] else 'N/A'}")
                    logging.error(f"   First entities_message: {df['entities_message'].iloc[0]}")

                logging.error(f"🔵 Returning df...")
                logging.error(f"   Final df shape: {df.shape}")
                return df

            logging.error(f"🔵 Calling process_data...")
            df = process_data(events)
            logging.error(f"   df returned, shape: {df.shape if df is not None else 'None'}")

            logging.error(f"🔵 Defining safe_value function...")
            def safe_value(value) -> str:
                if value is None:
                    return "N/A"
                text_val = str(value).strip()
                return text_val if text_val else "N/A"

            logging.error(f"🔵 Building events_by_id dictionary...")
            events_by_id = {
                str(row["id"]): row
                for _, row in df.iterrows()
                if str(row["id"]).strip()
            }
            logging.error(f"   events_by_id count: {len(events_by_id)}")
            logging.error(f"   events_by_id keys (first 5): {list(events_by_id.keys())[:5]}")
            
            logging.error(f"🔵 Calculating new_ids (not in existing_ids)...")
            logging.error(f"   existing_ids: {existing_ids}")
            new_ids = [eid for eid in events_by_id if eid not in existing_ids]
            logging.error(f"   new_ids count: {len(new_ids)}")
            logging.error(f"   new_ids (first 5): {new_ids[:5]}")

            # Generate report content
            logging.error(f"🔵 Generating report content...")
            logging.error(f"🔵 Getting timestamp...")
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            logging.error(f"   timestamp: '{timestamp}'")
            
            logging.error(f"🔵 Generating slug from query...")
            slug = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_") or "seq_query"
            logging.error(f"   slug: '{slug}'")
            
            logging.error(f"🔵 Building filename...")
            filename = f"{slug}_{timestamp.replace(' ', '_').replace(':', '')}.txt"
            logging.error(f"   filename: '{filename}'")

            logging.error(f"🔵 Building report_lines...")
            report_lines = [
                f"Seq Filter: {query}",
                f"Retrieved At (UTC): {timestamp}",
            ]
            if from_date:
                report_lines.append(f"Time Range Start: {from_date}")
            if to_date:
                report_lines.append(f"Time Range End: {to_date}")
            if from_date or to_date:
                report_lines.append("")
            report_lines.append(f"Total Events Returned: {len(events)}")
            report_lines.append("")
            logging.error(f"   report_lines count (initial): {len(report_lines)}")
            
            logging.error(f"🔵 Building archive_lines...")
            archive_lines = [
                f"Seq Filter: {query}",
                f"Retrieved At (UTC): {timestamp}",
            ]
            if from_date:
                archive_lines.append(f"Time Range Start: {from_date}")
            if to_date:
                archive_lines.append(f"Time Range End: {to_date}")
            if from_date or to_date:
                archive_lines.append("")
            archive_lines.append(f"New Events Archived: {len(new_ids)}")
            archive_lines.append("")
            logging.error(f"   archive_lines count (initial): {len(archive_lines)}")

            for index, eid in enumerate(events_by_id, start=1):
                event_lines = []
                row = events_by_id[eid]
                
                entities_source = row.get("entities_message", [])
                entities = (
                    list(entities_source)
                    if isinstance(entities_source, (list, tuple))
                    else []
                )
                
                event_lines.extend([
                    f"Event {index}",
                    f"Timestamp: {safe_value(row['timestamp'])}",
                    f"Level: {safe_value(row['level'])}",
                    f"Event ID: {safe_value(row['id'])}",
                    f"Message: {safe_value(row['message'])}",
                ])
                
                if row.get("exception"):
                    event_lines.append(f"Exception: {safe_value(row['exception'])}")
                
                if entities:
                    event_lines.append(f"Entities: {', '.join(entities)}")
                
                # Add properties if interesting? Maybe too verbose.
                # props = row.get("properties", {})
                # if props:
                #     event_lines.append(f"Properties: {json.dumps(props)}")
                
                event_lines.append("")
                
                report_lines.extend(event_lines)
                if eid in new_ids:
                    archive_lines.extend(event_lines)

            report_content = "\n".join(report_lines)
            archive_content = "\n".join(archive_lines)

            response_sections: List[str] = []

            # 4. Include Existing Knowledge (RAG)
            if existing_text:
                response_sections.append(f"**Existing Knowledge Base Records**:\n{existing_text}")

            # 5. Archive New Events
            logging.error(f"🔍 === ARCHIVE NEW EVENTS START ===")
            logging.error(f"🔵 Checking if new_ids exists...")
            logging.error(f"   new_ids count: {len(new_ids) if new_ids else 0}")
            if new_ids:
                logging.error(f"   ✅ new_ids has items")
                logging.error(f"🔵 Calling progress_update...")
                await emitter.progress_update(f"🆕 Found {len(new_ids)} new events; archiving update...")
                logging.error(f"   progress_update complete")
                
                # Ensure content is not empty and has minimum length
                logging.error(f"🔵 Checking archive_content...")
                logging.error(f"   archive_content length: {len(archive_content) if archive_content else 0}")
                logging.error(f"   archive_content (first 200 chars): {archive_content[:200] if archive_content else 'N/A'}")
                
                if not archive_content or len(archive_content.strip()) < 10:
                    logging.error(f"   ❌ archive_content is empty or too short")
                    logging.error(f"🔵 Calling error_update...")
                    await emitter.error_update("⚠️ Archive content is empty or too short, skipping knowledge base update")
                    logging.error(f"   error_update complete")
                else:
                    logging.error(f"   ✅ archive_content is valid")
                    try:
                        logging.error(f"🔵 Calling upload_report_file...")
                        logging.error(f"   filename: '{filename}'")
                        logging.error(f"   content length: {len(archive_content)}")
                        logging.error(f"   metadata: {{'source': '{query}', 'type': 'seq_report'}}")
                        
                        file_record = await KnowledgeRepository.upload_report_file(
                            request=request,
                            user=user,
                            filename=filename,
                            content=archive_content,
                            metadata={"source": query, "type": "seq_report"},
                        )
                        logging.error(f"   file_record returned")
                        logging.error(f"   file_record['id']: {file_record['id'] if file_record else 'None'}")
                        logging.error(f"   file_record keys: {file_record.keys() if file_record else 'None'}")
                        
                        logging.error(f"🔵 Calling attach_file_to_knowledge...")
                        logging.error(f"   kb_id: {kb.id}")
                        logging.error(f"   file_id: {file_record['id']}")
                        logging.error(f"   content length: {len(archive_content)}")
                        
                        await KnowledgeRepository.attach_file_to_knowledge(
                            request=request,
                            user=user,
                            kb_id=kb.id,
                            file_id=file_record["id"],
                            content=archive_content,
                        )
                        logging.error(f"   attach_file_to_knowledge complete")
                        
                        logging.error(f"🔵 Calling progress_update...")
                        await emitter.progress_update(f"✅ Saved {len(new_ids)} events to knowledge base")
                        logging.error(f"   progress_update complete")
                    except Exception as kb_error:
                        logging.error(f"   ❌ Exception caught: {kb_error}")
                        logging.error(f"   Exception type: {type(kb_error)}")
                        import traceback
                        logging.error(f"   Traceback: {traceback.format_exc()}")
                        logging.error(f"🔵 Calling error_update...")
                        await emitter.error_update(f"⚠️ Failed to save to knowledge base: {kb_error}")
                        logging.error(f"   error_update complete")
                        # Continue execution even if knowledge base update fails
                
                new_summaries = []
                for eid in new_ids:
                    row = events_by_id[eid]
                    new_summaries.append(
                        f"[{row['timestamp']}] {row['level']}: {row['message'][:100]}..."
                    )
                response_sections.append(
                    f"**New Records Archived ({timestamp})**:\n"
                    + ("\n".join(new_summaries) if new_summaries else "Stored new snapshot.")
                )
            else:
                await emitter.progress_update("ℹ️ No new events compared to stored history")
                response_sections.append("**Update Notice**: No new Seq events were found. Existing snapshot remains current.")

            # 6. Retrieve Updated RAG (only if we successfully saved to knowledge base)
            if new_ids and archive_content and len(archive_content.strip()) >= 10:
                try:
                    await emitter.progress_update("🔄 Retrieving latest knowledge snapshot...")
                    updated_rag = await KnowledgeRepository.query_knowledge_base(
                        request=request,
                        user=user,
                        kb_id=kb.id,
                        query=query,
                        limit=self.valves.max_events,
                        valves=self.valves,
                    )
                    updated_docs = updated_rag.get("documents", [])
                    if updated_docs and len(updated_docs) > 0 and updated_docs[0]:
                        combined = "\n\n".join(updated_docs[0])
                        if not existing_text or combined != existing_text:
                            response_sections.append(
                                f"**Updated Knowledge Snapshot**:\n{combined}"
                            )
                except Exception as rag_error:
                    await emitter.error_update(f"⚠️ Failed to retrieve updated knowledge: {rag_error}")
                    # Continue execution even if RAG retrieval fails

            await emitter.success_update("✅ Analysis complete!")

            # Build final result text with safe section joining
            sections_text = "\n\n".join(response_sections) if response_sections else "No response sections generated."
            
            result_text = (
                f"🔬 **Seq Log Analysis Results** — {query}\n\n"
                + sections_text + "\n\n"
                + "--- **Full Report of Current Search** ---\n"
                + "(This section contains the full text of the events found in this session)\n\n"
                + report_content
            )
            
            await eventer(
                {
                    "type": "result",
                    "data": {
                        "description": seq_debug + result_text,
                        "done": True,
                        "hidden": False,
                    },
                }
            )
            return seq_debug + result_text

        except Exception as exc:
            import traceback
            full_error = traceback.format_exc()
            await emitter.error_update(f"❌ Unexpected Error: {exc}")
            error_msg = f"❌ **Seq Tool Unexpected Error**: {exc}\n\n**Traceback**:\n```\n{full_error}\n```"
            await eventer(
                {
                    "type": "result",
                    "data": {"description": error_msg, "done": True, "hidden": False},
                }
            )
            return seq_debug + error_msg

    async def discover_seq_metadata(
        self,
        __event_emitter__=None,
    ) -> str:
        """
        🔍 Discover Seq Metadata (Signals, Properties, Values)
        
        Queries the Seq server to discover:
        - Available signals (e.g., Production, Errors, UAT)
        - Available properties (e.g., ENV, ROLE, @Level)
        - Distinct values for custom properties
        
        Use this FIRST to understand what you can query in analyze_seq_logs().
        """
        async def _noop_event(*_args, **_kwargs):
            return None

        eventer = __event_emitter__ if __event_emitter__ is not None else _noop_event
        emitter = EventEmitter(eventer)
        
        await emitter.progress_update("🔍 Discovering Seq metadata...")
        
        try:
            headers = {"X-Seq-ApiKey": self.valves.seq_api_key}
            base_url = self.valves.seq_server_url.rstrip('/')
            
            metadata_parts = []
            
            # 1. Discover Signals
            await emitter.progress_update("📡 Fetching signals...")
            try:
                signals_response = requests.get(f"{base_url}/api/signals", headers=headers, timeout=10)
                if signals_response.status_code == 200:
                    signals = signals_response.json()
                    if signals:
                        signal_lines = ["## 🎯 Available Signals\n"]
                        for sig in signals:
                            title = sig.get("Title", "Unknown")
                            sig_id = sig.get("Id", "")
                            desc = sig.get("Description", "")
                            filter_expr = sig.get("ExplicitFilter", "")
                            signal_lines.append(f"### {title}")
                            signal_lines.append(f"- **ID**: `{sig_id}`")
                            if desc:
                                signal_lines.append(f"- **Description**: {desc}")
                            if filter_expr:
                                signal_lines.append(f"- **Filter**: `{filter_expr}`")
                            signal_lines.append("")
                        metadata_parts.append("\n".join(signal_lines))
            except Exception as e:
                metadata_parts.append(f"⚠️ Could not fetch signals: {e}")
            
            # 2. Discover Properties
            await emitter.progress_update("📊 Fetching properties...")
            try:
                props_response = requests.get(f"{base_url}/api/expressions/properties", headers=headers, timeout=10)
                if props_response.status_code == 200:
                    properties = props_response.json()
                    if properties:
                        prop_lines = ["## 🏷️ Available Properties\n"]
                        
                        # Group by type
                        standard_props = [p for p in properties if p["Name"].startswith("@")]
                        custom_props = [p for p in properties if not p["Name"].startswith("@")]
                        
                        if standard_props:
                            prop_lines.append("### Standard Properties")
                            for prop in standard_props:
                                name = prop.get("Name")
                                prop_type = prop.get("Type", "Unknown")
                                desc = prop.get("Description", "")
                                line = f"- `{name}` ({prop_type})"
                                if desc:
                                    line += f": {desc}"
                                prop_lines.append(line)
                            prop_lines.append("")
                        
                        if custom_props:
                            prop_lines.append("### Custom Properties")
                            for prop in custom_props:
                                name = prop.get("Name")
                                prop_type = prop.get("Type", "Unknown")
                                prop_lines.append(f"- `{name}` ({prop_type})")
                            prop_lines.append("")
                        
                        metadata_parts.append("\n".join(prop_lines))
                        
                        # 3. Get distinct values for key custom properties
                        await emitter.progress_update("🔢 Fetching distinct values...")
                        distinct_lines = ["## 📋 Distinct Values for Custom Properties\n"]
                        important_props = [p["Name"] for p in custom_props if p["Name"] in ["ENV", "ROLE", "Application", "SLOT", "Environment"]]
                        
                        for prop_name in important_props:
                            try:
                                distinct_response = requests.get(
                                    f"{base_url}/api/expressions/distinct",
                                    headers=headers,
                                    params={"property": prop_name, "take": 50},
                                    timeout=10
                                )
                                if distinct_response.status_code == 200:
                                    values = distinct_response.json().get("Values", [])
                                    if values:
                                        distinct_lines.append(f"### {prop_name}")
                                        distinct_lines.append(", ".join(f"`{v}`" for v in values[:20]))
                                        distinct_lines.append("")
                            except Exception:
                                pass
                        
                        if len(distinct_lines) > 1:
                            metadata_parts.append("\n".join(distinct_lines))
                        
            except Exception as e:
                metadata_parts.append(f"⚠️ Could not fetch properties: {e}")
            
            # 4. Usage Examples
            examples = [
                "## 💡 Query Examples\n",
                "### Natural Language (Recommended)",
                "- \"Show me errors in production from the last 24 hours\"",
                "- \"Find all warnings in QA environment\"",
                "- \"Analyze slow requests in CD role\"",
                "- \"Show me STAGE errors for CM role\"",
                "",
                "### Seq Filter Syntax",
                "- `@Level == 'ERROR' && ENV == 'PROD'` (use UPPERCASE for levels)",
                "- `ROLE == 'CD' && Duration > 1000`",
                "- `ENV == 'QA' && ROLE == 'CM'`",
                "- `@Level == 'WARNING' && ENV == 'STAGE'`",
                "- `@Level == 'INFO'` (valid levels: ERROR, WARNING, INFO)",
                "",
                "### With Signals",
                "- Use `signal='Production'` parameter",
                "- Use `signal='signal-123'` parameter with signal ID",
                "",
                "### Valid Values",
                "- **ENV**: `PROD`, `QA`, `STAGE`",
                "- **ROLE**: `CD`, `CM`",
            ]
            metadata_parts.append("\n".join(examples))
            
            await emitter.success_update("✅ Metadata discovery complete!")
            
            result = "# 🔍 Seq Metadata Discovery\n\n" + "\n\n".join(metadata_parts)
            
            await eventer({
                "type": "result",
                "data": {"description": result, "done": True, "hidden": False}
            })
            
            return result
            
        except Exception as exc:
            error_msg = f"❌ **Metadata Discovery Error**: {exc}"
            await emitter.error_update(error_msg)
            await eventer({
                "type": "result",
                "data": {"description": error_msg, "done": True, "hidden": False}
            })
            return error_msg


class KnowledgeRepository:
    """Helper for resolving OpenWebUI knowledge bases without exposing extra tool methods."""

    @staticmethod
    async def load_by_user(
        user_id: Any, permission: str = "write"
    ) -> List[KnowledgeUserModel]:
        knowledge = await run_in_threadpool(
            Knowledges.get_knowledge_bases_by_user_id, user_id, permission
        )
        return knowledge or []

    @staticmethod
    async def find_by_name(
        user_id: Any, identifier: str, permission: str = "write"
    ) -> Optional[KnowledgeUserModel]:
        knowledge_bases = await KnowledgeRepository.load_by_user(user_id, permission)
        if not knowledge_bases:
            return None

        normalized = identifier.strip().lower()
        id_lookup = {kb.id: kb for kb in knowledge_bases}
        if identifier in id_lookup:
            return id_lookup[identifier]

        by_name = {
            (kb.name or "").strip().lower(): kb for kb in knowledge_bases if kb.name
        }
        return by_name.get(normalized)

    @staticmethod
    def dedupe_preserve_order(values: List[Any]) -> List[Any]:
        seen: Set[str] = set()
        unique: List[Any] = []
        for value in values:
            marker = repr(value)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(value)
        return unique

    @staticmethod
    async def resolve_user(__user__: Optional[dict]) -> Any:
        if not __user__ or not __user__.get("id"):
            raise ValueError("User context with an 'id' is required")
        user = await run_in_threadpool(Users.get_user_by_id, str(__user__["id"]))
        if not user:
            raise ValueError("Unable to resolve OpenWebUI user")
        return user

    @staticmethod
    def require_request(__request__: Optional[Any]) -> Any:
        if __request__ is None or not isinstance(__request__, Request):
            raise ValueError("Request context is required inside OpenWebUI")
        return __request__

    @staticmethod
    async def query_knowledge_base(
        request: Any,
        user: Any,
        kb_id: str,
        query: str,
        limit: int,
        valves: Optional[Any] = None,
    ) -> Dict[str, Any]:
        max_results = getattr(valves, "max_events", limit) if valves else limit
        form_kwargs: Dict[str, Any] = {
            "collection_names": [kb_id],
            "query": query,
            "k": max(limit, max_results),
            "hybrid": getattr(valves, "enable_hybrid_search", False) if valves else False,
        }
        reranker = getattr(valves, "reranker_results", 0) if valves else 0
        if reranker > 0:
            form_kwargs["k_reranker"] = reranker
        relevance = getattr(valves, "relevance_threshold", 0.0) if valves else 0.0
        if relevance > 0:
            form_kwargs["r"] = relevance
        form = QueryCollectionsForm(**form_kwargs)
        return await query_collection_handler(request=request, form_data=form, user=user)

    @staticmethod
    async def upload_report_file(
        request: Any,
        user: Any,
        filename: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        safe_metadata = metadata.copy() if metadata else {}
        safe_metadata.setdefault("source", "seq_tool")
        safe_metadata.setdefault("type", "text")
        
        final_metadata = {}
        for k, v in safe_metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                final_metadata[k] = v
            else:
                final_metadata[k] = str(v)

        upload = UploadFile(
            filename=filename,
            file=SpooledTemporaryFile(max_size=1024 * 1024),
            headers={"content-type": "text/plain"},
        )
        upload.file.write(content.encode("utf-8"))
        upload.file.seek(0)
        try:
            # upload_file_handler can be async in some OpenWebUI versions
            if inspect.iscoroutinefunction(upload_file_handler):
                result = await upload_file_handler(
                    request,
                    upload,
                    final_metadata,
                    False,  # defer processing; process once in attach_file_to_knowledge
                    False,
                    user,
                    None,
                )
            else:
                result = await run_in_threadpool(
                    upload_file_handler,
                    request,
                    upload,
                    final_metadata,
                    False,  # defer processing; process once in attach_file_to_knowledge
                    False,
                    user,
                    None,
                )
        finally:
            await upload.close()
        
        file_id = getattr(result, "id", None)
        if file_id is None and isinstance(result, dict):
            file_id = result.get("id")

        if not file_id:
            raise ValueError("Failed to upload report content into OpenWebUI files")

        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        return result

    @staticmethod
    async def attach_file_to_knowledge(
        request: Any,
        user: Any,
        kb_id: str,
        file_id: str,
        content: Optional[str] = None,
    ) -> None:
        # 1. Associate file with knowledge base
        try:
            await run_in_threadpool(
                Knowledges.add_file_to_knowledge_by_id,
                kb_id,
                file_id,
                user.id,
            )
        except AttributeError:
            # Fallback for older versions or if method missing
            def _update_metadata() -> bool:
                knowledge = Knowledges.get_knowledge_by_id(id=kb_id)
                if not knowledge:
                    return False
                data = getattr(knowledge, "data", None) or {}
                file_ids = data.get("file_ids", [])
                if file_id not in file_ids:
                    file_ids.append(file_id)
                    data["file_ids"] = file_ids
                    Knowledges.update_knowledge_data_by_id(id=kb_id, data=data)
                return True

            updated = await run_in_threadpool(_update_metadata)
            if not updated:
                raise ValueError("Failed to update knowledge metadata with new file")

        # 2. Process the file into the knowledge collection synchronously
        try:
            form = ProcessFileForm(file_id=file_id, collection_name=kb_id, content=content)
            if inspect.iscoroutinefunction(process_file):
                await process_file(request=request, form_data=form, user=user)
            else:
                await run_in_threadpool(process_file, request, form, user)
        except Exception as exc:
            raise ValueError(f"Failed to process file into knowledge base: {exc}") from exc
