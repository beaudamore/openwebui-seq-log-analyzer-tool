# OpenWebUI Seq Log Analyzer Tool

Advanced log analysis tool for Seq (Structured Event Query) with native OpenWebUI integrations, NLP-powered insights, and intelligent query processing.

**Version:** 1.0.0  
**Author:** Beau D'Amore ([www.damore.ai](https://www.damore.ai))

## Features

- **Native OpenWebUI Integration**:
  - Direct Seq log queries from chat
  - Knowledge base storage for historical analysis
  - File upload support for offline log analysis
  - RAG-powered log insights

- **Intelligent Query Processing**:
  - Natural language query interpretation
  - Automatic date range parsing ("yesterday", "last week", "past 24 hours")
  - Query validation and optimization
  - Signal and property auto-discovery

- **NLP-Powered Analysis**:
  - Entity extraction from log messages
  - Keyword and pattern detection
  - Sentiment analysis
  - Automatic categorization

- **Rich Log Analysis**:
  - Error pattern detection
  - Anomaly identification
  - Trend analysis over time
  - User agent and request analysis
  - Performance metrics extraction

- **Knowledge Base Features**:
  - Automatic log embedding for RAG queries
  - Historical log search
  - Cross-query correlation
  - Deduplication handling

- **Metadata Discovery**:
  - Available signal discovery
  - Property enumeration
  - Value sampling for filters
  - Query hints and suggestions

## Installation

### Requirements

```bash
pip install fastapi pandas spacy nltk requests
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords
```

### Setup in OpenWebUI

1. Upload `tool/seq_log_tool.py` to OpenWebUI
2. Configure valves:
   - Set Seq server URL
   - Add API key
   - Configure knowledge base name
   - Adjust analysis settings

## Usage

### Query Examples

```
"Show me errors from yesterday"
"List unique user agents in the last 24 hours"
"Find all 500 errors in production"
"What happened between 2pm and 4pm today?"
"Show me slow requests over 5 seconds"
"Analyze log patterns for user login failures"
```

### Natural Language Features

The tool automatically interprets:
- Relative dates: "yesterday", "last week", "past hour"
- Time ranges: "between 2pm and 4pm"
- Error levels: "errors", "warnings", "critical"
- Environments: "production", "staging", "development"
- Common patterns: "slow requests", "failed logins", "exceptions"

## Configuration (Valves)

### Seq Connection

- **seq_url**: Seq server URL (e.g., `http://localhost:5341`)
- **seq_api_key**: Seq API key for authentication
- **default_signal**: Default signal to query (if not specified)

### Analysis Settings

- **default_knowledge_base**: KB name for log storage (default: `Seq Logs`)
- **max_results**: Maximum log entries to return (default: 100)
- **enable_nlp_analysis**: Enable NLP entity extraction
- **enable_embedding**: Store logs in knowledge base
- **enable_hybrid_search**: Use semantic + keyword search for KB queries

### Display Options

- **enable_debug**: Show query parameters and metadata
- **render_mode**: Format for results (table, json, compact)
- **include_metadata**: Include Seq metadata in results

## Advanced Features

### Metadata Discovery

Use `discover_seq_metadata()` to:
- List available signals in your Seq instance
- Discover queryable properties
- Sample property values for building filters
- Get query syntax hints

### Knowledge Base Integration

Logs can be automatically embedded for:
- Historical trend analysis
- Pattern correlation
- RAG-powered insights
- Cross-timeframe queries

### File Upload Support

Upload Seq export files (.json, .csv) for offline analysis:
- Bulk log processing
- Historical analysis without Seq access
- Correlation with live data

## System Prompt

The tool includes an intelligent system prompt (see `prompt/seq-analyzer-prompt.md`) that ensures:
- Immediate execution of queries (no explanations unless asked)
- Natural language interpretation
- Automatic tool selection
- Context-aware responses

## Documentation

- [System Prompt](prompt/seq-analyzer-prompt.md)

## Seq Information

This tool works with Seq, a structured log server:
- Website: https://datalust.co/seq
- Documentation: https://docs.datalust.co/
- API Reference: https://docs.datalust.co/reference/api-overview

## License

MIT License - See individual repository for details

## Author

**Beau D'Amore**  
[www.damore.ai](https://www.damore.ai)
