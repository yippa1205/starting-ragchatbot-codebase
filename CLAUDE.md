# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

**Development Server:**
```bash
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Dependencies:**
```bash
uv sync  # Install Python dependencies
```

**Environment Setup:**
Create `.env` file in root with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot system** that answers questions about course materials. The system uses a modular architecture with clear separation between document processing, vector storage, AI generation, and web interface.

### Core Data Flow

1. **Document Processing Pipeline** (`document_processor.py`):
   - Parses course documents with specific format: Course Title/Link/Instructor → Lessons with content
   - Implements sentence-based chunking with configurable overlap (800 chars, 100 overlap)
   - Creates `CourseChunk` objects with course/lesson metadata

2. **Vector Storage Layer** (`vector_store.py`):
   - ChromaDB for persistent vector storage with sentence-transformer embeddings
   - Dual collection strategy: course metadata + content chunks
   - Unified search interface supporting course name and lesson number filtering

3. **RAG Orchestration** (`rag_system.py`):
   - Central coordinator managing all components
   - Tool-based architecture with `CourseSearchTool` for semantic search
   - Session management for conversation continuity

4. **AI Generation** (`ai_generator.py`):
   - Anthropic Claude integration with tool calling capabilities
   - System prompt optimized for educational content responses
   - Two-phase processing: tool execution → final response generation

### Key Architectural Patterns

**Tool-Based Search Architecture:**
- `ToolManager` registers and executes tools dynamically
- `CourseSearchTool` implements semantic search with intelligent course name matching
- Sources tracked and returned to frontend for citation

**Session Management:**
- In-memory session storage with configurable history limits (default: 2 exchanges)
- Conversation context passed to AI for coherent multi-turn interactions
- Session IDs managed transparently between frontend/backend

**Component Configuration:**
All settings centralized in `config.py` using environment variables and dataclass:
- `CHUNK_SIZE`: 800 (document chunking)
- `CHUNK_OVERLAP`: 100 (chunk overlap)
- `MAX_RESULTS`: 5 (search results)
- `MAX_HISTORY`: 2 (conversation turns)
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"

## Critical Implementation Details

**Document Format Requirements:**
Course documents must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[lesson content...]
```

**Vector Storage Collections:**
- `course_metadata`: High-level course information for broad searches
- `course_content`: Chunked lesson content with embeddings
- Both collections use same embedding model for semantic consistency

**Frontend-Backend Integration:**
- RESTful API with `/api/query` (POST) and `/api/courses` (GET)
- Frontend handles session state, markdown rendering, and source display
- Loading states and error handling implemented client-side

**Tool Execution Flow:**
1. User query → RAG system → AI generator with tools
2. Claude decides whether to use search tool based on query type
3. If searching: tool execution → results formatting → final response
4. Sources captured and returned separately for UI display

## Development Notes

**Adding New Tools:**
1. Inherit from `Tool` base class in `search_tools.py`
2. Implement `get_tool_definition()` and `execute()` methods
3. Register with `ToolManager` in `RAG system.__init__()`

**Document Processing:**
- Supports .pdf, .docx, .txt files
- Chunking preserves sentence boundaries for better context
- Lesson numbering extracted via regex: `Lesson \d+:`

**Configuration Changes:**
Modify `config.py` for system-wide settings. Key parameters:
- Embedding model changes require ChromaDB rebuild
- Chunk size affects memory usage and search granularity
- History limits impact conversation context length
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies