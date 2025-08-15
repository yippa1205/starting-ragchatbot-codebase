import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the backend directory to Python path so we can import modules
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem

@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_config(temp_chroma_db):
    """Create test configuration with temporary database"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_db
    config.ANTHROPIC_API_KEY = "test_key"
    config.MAX_RESULTS = 3  # Smaller for testing
    return config

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction to Python",
            lesson_link="http://example.com/lesson1"
        ),
        Lesson(
            lesson_number=2,
            title="Variables and Data Types", 
            lesson_link="http://example.com/lesson2"
        )
    ]
    
    return Course(
        title="Python Basics",
        course_link="http://example.com/course",
        instructor="John Doe",
        lessons=lessons
    )

@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    
    # Create sample content for each lesson
    lesson_contents = {
        1: "Python is a programming language. It's easy to learn and powerful. You can build web apps, data science tools, and automation scripts.",
        2: "Variables store data in Python. Python has strings, integers, floats, and booleans. You can assign values using the equals sign."
    }
    
    for lesson in sample_course.lessons:
        lesson_content = lesson_contents.get(lesson.lesson_number, "Default lesson content")
        
        # Split content into smaller chunks for testing
        words = lesson_content.split()
        chunk_size = 10  # Small chunks for testing
        
        for i in range(0, len(words), chunk_size):
            chunk_content = " ".join(words[i:i+chunk_size])
            chunk = CourseChunk(
                course_title=sample_course.title,
                lesson_number=lesson.lesson_number,
                chunk_index=len(chunks),
                content=chunk_content
            )
            chunks.append(chunk)
    
    return chunks

@pytest.fixture
def vector_store(test_config):
    """Create a VectorStore instance for testing"""
    return VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )

@pytest.fixture
def populated_vector_store(vector_store, sample_course, sample_course_chunks):
    """Create a VectorStore populated with test data"""
    vector_store.add_course_metadata(sample_course)
    vector_store.add_course_content(sample_course_chunks)
    return vector_store

@pytest.fixture
def course_search_tool(vector_store):
    """Create a CourseSearchTool instance"""
    return CourseSearchTool(vector_store)

@pytest.fixture
def mock_ai_generator():
    """Create a mock AI generator for testing"""
    mock_generator = Mock(spec=AIGenerator)
    return mock_generator

@pytest.fixture
def tool_manager(course_search_tool):
    """Create a ToolManager with registered CourseSearchTool"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager

@pytest.fixture
def rag_system(test_config):
    """Create a RAG system instance for testing"""
    with patch('ai_generator.anthropic.Anthropic'):
        return RAGSystem(test_config)