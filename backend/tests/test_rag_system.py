import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from models import Course, Lesson


class TestRAGSystem:
    """Test suite for RAGSystem integration functionality"""
    
    def test_init_creates_all_components(self, test_config):
        """Test that RAG system initializes all required components"""
        with patch('ai_generator.anthropic.Anthropic'):
            rag = RAGSystem(test_config)
            
            assert rag.document_processor is not None
            assert rag.vector_store is not None
            assert rag.ai_generator is not None
            assert rag.session_manager is not None
            assert rag.tool_manager is not None
            assert rag.search_tool is not None
            
            # Verify tool is registered
            assert "search_course_content" in rag.tool_manager.tools
    
    def test_add_course_document_success(self, rag_system, tmp_path):
        """Test adding a course document successfully"""
        # Create a test document file
        test_file = tmp_path / "test_course.txt"
        test_content = """Course Title: Test Course
Course Link: http://example.com/course
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: http://example.com/lesson1
This is lesson 1 content about basics.

Lesson 2: Advanced Topics
Lesson Link: http://example.com/lesson2
This is lesson 2 content about advanced topics.
"""
        test_file.write_text(test_content)
        
        course, chunk_count = rag_system.add_course_document(str(test_file))
        
        assert course is not None
        assert course.title == "Test Course"
        assert course.instructor == "Test Instructor"
        assert len(course.lessons) == 2
        assert chunk_count > 0
        
        # Verify course was added to vector store
        titles = rag_system.vector_store.get_existing_course_titles()
        assert "Test Course" in titles
    
    def test_add_course_document_error(self, rag_system):
        """Test error handling when adding invalid document"""
        course, chunk_count = rag_system.add_course_document("nonexistent_file.txt")
        
        assert course is None
        assert chunk_count == 0
    
    def test_add_course_folder_success(self, rag_system, tmp_path):
        """Test adding multiple course documents from a folder"""
        # Create test files
        course1_file = tmp_path / "course1.txt"
        course1_content = """Course Title: Course One
Course Link: http://example.com/course1
Course Instructor: Instructor One

Lesson 1: Intro
Lesson Link: http://example.com/lesson1
Content for course one.
"""
        course1_file.write_text(course1_content)
        
        course2_file = tmp_path / "course2.txt"
        course2_content = """Course Title: Course Two
Course Link: http://example.com/course2
Course Instructor: Instructor Two

Lesson 1: Basics
Lesson Link: http://example.com/lesson1
Content for course two.
"""
        course2_file.write_text(course2_content)
        
        total_courses, total_chunks = rag_system.add_course_folder(str(tmp_path))
        
        assert total_courses == 2
        assert total_chunks > 0
        
        # Verify both courses were added
        titles = rag_system.vector_store.get_existing_course_titles()
        assert "Course One" in titles
        assert "Course Two" in titles
    
    def test_add_course_folder_with_clear_existing(self, rag_system, tmp_path, sample_course):
        """Test adding course folder with clear_existing=True"""
        # First add a course to the system
        rag_system.vector_store.add_course_metadata(sample_course)
        assert rag_system.vector_store.get_course_count() == 1
        
        # Create a new course file
        new_course_file = tmp_path / "new_course.txt"
        new_course_content = """Course Title: New Course
Course Link: http://example.com/new
Course Instructor: New Instructor

Lesson 1: New Lesson
Lesson Link: http://example.com/lesson1
New lesson content.
"""
        new_course_file.write_text(new_course_content)
        
        # Add folder with clear_existing=True
        total_courses, total_chunks = rag_system.add_course_folder(
            str(tmp_path), 
            clear_existing=True
        )
        
        assert total_courses == 1
        assert total_chunks > 0
        
        # Verify old course was cleared and only new course exists
        titles = rag_system.vector_store.get_existing_course_titles()
        assert "New Course" in titles
        assert "Python Basics" not in titles
        assert rag_system.vector_store.get_course_count() == 1
    
    def test_add_course_folder_skip_existing(self, rag_system, tmp_path, sample_course):
        """Test that existing courses are skipped when adding folder"""
        # First add a course to the system
        rag_system.vector_store.add_course_metadata(sample_course)
        
        # Create a file with the same course title
        existing_course_file = tmp_path / "existing.txt"
        existing_course_content = """Course Title: Python Basics
Course Link: http://example.com/duplicate
Course Instructor: Different Instructor

Lesson 1: Duplicate
Lesson Link: http://example.com/dup
Duplicate content.
"""
        existing_course_file.write_text(existing_course_content)
        
        total_courses, total_chunks = rag_system.add_course_folder(str(tmp_path))
        
        # Should not add the duplicate course
        assert total_courses == 0
        assert total_chunks == 0
        assert rag_system.vector_store.get_course_count() == 1
    
    def test_add_course_folder_nonexistent(self, rag_system):
        """Test adding course folder that doesn't exist"""
        total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/folder")
        
        assert total_courses == 0
        assert total_chunks == 0
    
    def test_query_without_session(self, rag_system):
        """Test querying without session ID"""
        # Mock the AI generator's response directly
        rag_system.ai_generator.generate_response = Mock(return_value="Test response")
        
        response, sources = rag_system.query("What is Python?")
        
        assert response == "Test response"
        assert sources == []  # No sources since no search was performed
        
        # Verify AI generator was called with correct parameters
        call_args = rag_system.ai_generator.generate_response.call_args
        assert "Answer this question about course materials: What is Python?" in call_args[1]["query"]
        assert call_args[1]["tools"] is not None
    
    def test_query_with_session(self, rag_system):
        """Test querying with session management"""
        # Mock the AI generator's response directly
        rag_system.ai_generator.generate_response = Mock(return_value="Response with session")
        
        session_id = "test_session_123"
        response, sources = rag_system.query("How are you?", session_id=session_id)
        
        assert response == "Response with session"
        
        # Verify session was updated with the exchange
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "How are you?" in history
        assert "Response with session" in history
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_with_tool_execution(self, mock_anthropic, populated_vector_store, test_config):
        """Test querying that triggers tool execution"""
        # Create RAG system with populated data
        rag = RAGSystem(test_config)
        rag.vector_store = populated_vector_store  # Use populated store
        rag.search_tool = type(rag.search_tool)(populated_vector_store)  # Recreate tool with populated store
        rag.tool_manager.register_tool(rag.search_tool)
        
        mock_client = mock_anthropic.return_value
        
        # Mock tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "Python programming"}
        mock_tool_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_content_block = Mock()
        mock_final_content_block.text = "Python is a programming language used for development."
        mock_final_response.content = [mock_final_content_block]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        response, sources = rag.query("Tell me about Python programming")
        
        assert response == "Python is a programming language used for development."
        assert len(sources) > 0
        assert "Python Basics" in sources[0]
        
        # Verify tool was executed
        assert mock_client.messages.create.call_count == 2
    
    def test_get_course_analytics_empty(self, rag_system):
        """Test getting analytics from empty system"""
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []
    
    def test_get_course_analytics_populated(self, rag_system, sample_course):
        """Test getting analytics from populated system"""
        rag_system.vector_store.add_course_metadata(sample_course)
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 1
        assert "Python Basics" in analytics["course_titles"]
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_error_handling(self, mock_anthropic, rag_system):
        """Test query error handling when AI generator fails"""
        # Mock the AI generator directly instead of the client
        rag_system.ai_generator.generate_response = Mock(side_effect=Exception("API Error"))
        
        with pytest.raises(Exception, match="API Error"):
            rag_system.query("test query")
    
    def test_tool_manager_integration(self, rag_system):
        """Test that tool manager is properly integrated"""
        # Verify search tool is registered
        tools = rag_system.tool_manager.get_tool_definitions()
        assert len(tools) == 1
        assert tools[0]["name"] == "search_course_content"
        
        # Test tool execution through manager
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", 
            query="test query"
        )
        
        # Should return "no results" for empty store
        assert "No relevant content found" in result
    
    def test_session_manager_integration(self, rag_system):
        """Test session manager integration"""
        session_id = "test_session"
        
        # Initially no history
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history is None
        
        # Add an exchange
        rag_system.session_manager.add_exchange(session_id, "Question", "Answer")
        
        # Should now have history
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "Question" in history
        assert "Answer" in history