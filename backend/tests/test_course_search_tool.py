import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool functionality"""
    
    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is properly formatted for Anthropic API"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_with_empty_vector_store(self, course_search_tool):
        """Test execute with empty vector store - should return no results"""
        result = course_search_tool.execute("What is Python?")
        
        assert "No relevant content found" in result
        assert course_search_tool.last_sources == []
    
    def test_execute_with_populated_vector_store(self, populated_vector_store):
        """Test execute with populated vector store - should return results"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("What is Python?")
        
        # Should not be a "no results" message
        assert "No relevant content found" not in result
        # Should contain formatted results with course info
        assert "[Python Basics" in result
        # Should have sources tracked
        assert len(tool.last_sources) > 0
    
    def test_execute_with_course_name_filter(self, populated_vector_store):
        """Test execute with course name filtering"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("programming", course_name="Python Basics")
        
        # Should find results since we have Python Basics course
        assert "No relevant content found" not in result
        assert "[Python Basics" in result
    
    def test_execute_with_nonexistent_course_filter(self, populated_vector_store):
        """Test execute with non-existent course name filter"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("programming", course_name="Non-existent Course")
        
        # Should return error for non-existent course
        assert "No course found matching 'Non-existent Course'" in result
    
    def test_execute_with_lesson_number_filter(self, populated_vector_store):
        """Test execute with lesson number filtering"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("variables", lesson_number=2)
        
        # Should find results from lesson 2 about variables
        assert "No relevant content found" not in result
        assert "Lesson 2" in result
    
    def test_execute_with_combined_filters(self, populated_vector_store):
        """Test execute with both course name and lesson number filters"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("variables", course_name="Python Basics", lesson_number=2)
        
        # Should find results from Python Basics lesson 2
        assert "No relevant content found" not in result
        assert "[Python Basics - Lesson 2]" in result
    
    def test_execute_with_vector_store_error(self, course_search_tool):
        """Test execute when vector store returns an error"""
        # Mock the vector store to return an error
        error_results = SearchResults.empty("Database connection failed")
        course_search_tool.store.search = Mock(return_value=error_results)
        
        result = course_search_tool.execute("test query")
        
        assert result == "Database connection failed"
        assert course_search_tool.last_sources == []
    
    def test_format_results_with_metadata(self, course_search_tool):
        """Test the _format_results method with proper metadata"""
        # Create mock search results
        results = SearchResults(
            documents=["Python is great", "Variables are important"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        
        formatted = course_search_tool._format_results(results)
        
        assert "[Python Basics - Lesson 1]" in formatted
        assert "[Python Basics - Lesson 2]" in formatted
        assert "Python is great" in formatted
        assert "Variables are important" in formatted
        
        # Check sources are tracked
        expected_sources = ["Python Basics - Lesson 1", "Python Basics - Lesson 2"]
        assert course_search_tool.last_sources == expected_sources
    
    def test_format_results_without_lesson_number(self, course_search_tool):
        """Test _format_results with missing lesson number in metadata"""
        results = SearchResults(
            documents=["General course info"],
            metadata=[{"course_title": "Python Basics"}],  # No lesson_number
            distances=[0.1]
        )
        
        formatted = course_search_tool._format_results(results)
        
        assert "[Python Basics]" in formatted  # No lesson number in header
        assert "General course info" in formatted
        assert course_search_tool.last_sources == ["Python Basics"]


class TestToolManager:
    """Test suite for ToolManager functionality"""
    
    def test_register_tool(self, course_search_tool):
        """Test registering a tool with the manager"""
        manager = ToolManager()
        manager.register_tool(course_search_tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == course_search_tool
    
    def test_get_tool_definitions(self, tool_manager):
        """Test getting tool definitions for API"""
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool_success(self, tool_manager, populated_vector_store):
        """Test successful tool execution"""
        # Register a tool with populated data
        search_tool = CourseSearchTool(populated_vector_store)
        tool_manager.register_tool(search_tool)
        
        result = tool_manager.execute_tool("search_course_content", query="Python")
        
        assert "No relevant content found" not in result
        assert "[Python Basics" in result
    
    def test_execute_tool_not_found(self, tool_manager):
        """Test executing a non-existent tool"""
        result = tool_manager.execute_tool("non_existent_tool", query="test")
        
        assert result == "Tool 'non_existent_tool' not found"
    
    def test_get_last_sources(self, tool_manager, populated_vector_store):
        """Test retrieving sources from last search"""
        # Execute a search that should generate sources
        search_tool = CourseSearchTool(populated_vector_store)
        tool_manager.register_tool(search_tool)
        
        tool_manager.execute_tool("search_course_content", query="Python")
        sources = tool_manager.get_last_sources()
        
        assert len(sources) > 0
        assert "Python Basics" in sources[0]
    
    def test_reset_sources(self, tool_manager, populated_vector_store):
        """Test resetting sources after retrieval"""
        # Execute a search that should generate sources
        search_tool = CourseSearchTool(populated_vector_store)
        tool_manager.register_tool(search_tool)
        
        tool_manager.execute_tool("search_course_content", query="Python")
        assert len(tool_manager.get_last_sources()) > 0
        
        tool_manager.reset_sources()
        assert tool_manager.get_last_sources() == []