import pytest
from unittest.mock import Mock, patch
from vector_store import VectorStore, SearchResults


class TestVectorStore:
    """Test suite for VectorStore functionality"""
    
    def test_init_creates_collections(self, vector_store):
        """Test that VectorStore initializes with proper collections"""
        assert vector_store.course_catalog is not None
        assert vector_store.course_content is not None
        assert vector_store.max_results == 3  # From test config
    
    def test_add_course_metadata(self, vector_store, sample_course):
        """Test adding course metadata to the catalog"""
        vector_store.add_course_metadata(sample_course)
        
        # Verify course was added by checking existing titles
        titles = vector_store.get_existing_course_titles()
        assert sample_course.title in titles
    
    def test_add_course_content(self, vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        vector_store.add_course_content(sample_course_chunks)
        
        # Try to search for content to verify it was added
        results = vector_store.search("Python")
        assert not results.is_empty()
    
    def test_search_empty_store(self, vector_store):
        """Test searching in empty vector store"""
        results = vector_store.search("test query")
        
        assert results.is_empty()
        assert len(results.documents) == 0
        assert len(results.metadata) == 0
    
    def test_search_populated_store(self, populated_vector_store):
        """Test searching in populated vector store"""
        results = populated_vector_store.search("Python")
        
        assert not results.is_empty()
        assert len(results.documents) > 0
        assert len(results.metadata) > 0
        assert len(results.distances) > 0
        
        # Verify metadata contains expected fields
        for meta in results.metadata:
            assert "course_title" in meta
            assert "lesson_number" in meta
            assert "chunk_index" in meta
    
    def test_search_with_course_name_filter(self, populated_vector_store):
        """Test searching with course name filter"""
        results = populated_vector_store.search("programming", course_name="Python Basics")
        
        assert not results.is_empty()
        # All results should be from Python Basics course
        for meta in results.metadata:
            assert meta["course_title"] == "Python Basics"
    
    def test_search_with_nonexistent_course_filter(self, populated_vector_store):
        """Test searching with non-existent course name filter"""
        results = populated_vector_store.search("programming", course_name="Non-existent Course")
        
        assert results.error is not None
        assert "No course found matching" in results.error
    
    def test_search_with_lesson_number_filter(self, populated_vector_store):
        """Test searching with lesson number filter"""
        results = populated_vector_store.search("data", lesson_number=2)
        
        if not results.is_empty():
            # All results should be from lesson 2
            for meta in results.metadata:
                assert meta["lesson_number"] == 2
    
    def test_search_with_combined_filters(self, populated_vector_store):
        """Test searching with both course name and lesson number filters"""
        results = populated_vector_store.search(
            "variables", 
            course_name="Python Basics", 
            lesson_number=2
        )
        
        if not results.is_empty():
            for meta in results.metadata:
                assert meta["course_title"] == "Python Basics"
                assert meta["lesson_number"] == 2
    
    def test_resolve_course_name_exact_match(self, populated_vector_store):
        """Test course name resolution with exact match"""
        resolved_title = populated_vector_store._resolve_course_name("Python Basics")
        assert resolved_title == "Python Basics"
    
    def test_resolve_course_name_partial_match(self, populated_vector_store):
        """Test course name resolution with partial match"""
        resolved_title = populated_vector_store._resolve_course_name("Python")
        assert resolved_title == "Python Basics"
    
    def test_resolve_course_name_no_match(self, populated_vector_store):
        """Test course name resolution with no match"""
        resolved_title = populated_vector_store._resolve_course_name("JavaScript")
        assert resolved_title is None
    
    def test_build_filter_no_filters(self, vector_store):
        """Test filter building with no filters"""
        filter_dict = vector_store._build_filter(None, None)
        assert filter_dict is None
    
    def test_build_filter_course_only(self, vector_store):
        """Test filter building with course name only"""
        filter_dict = vector_store._build_filter("Python Basics", None)
        assert filter_dict == {"course_title": "Python Basics"}
    
    def test_build_filter_lesson_only(self, vector_store):
        """Test filter building with lesson number only"""
        filter_dict = vector_store._build_filter(None, 2)
        assert filter_dict == {"lesson_number": 2}
    
    def test_build_filter_combined(self, vector_store):
        """Test filter building with both course and lesson"""
        filter_dict = vector_store._build_filter("Python Basics", 2)
        expected = {
            "$and": [
                {"course_title": "Python Basics"},
                {"lesson_number": 2}
            ]
        }
        assert filter_dict == expected
    
    def test_get_existing_course_titles_empty(self, vector_store):
        """Test getting course titles from empty store"""
        titles = vector_store.get_existing_course_titles()
        assert titles == []
    
    def test_get_existing_course_titles_populated(self, populated_vector_store):
        """Test getting course titles from populated store"""
        titles = populated_vector_store.get_existing_course_titles()
        assert "Python Basics" in titles
        assert len(titles) == 1
    
    def test_get_course_count_empty(self, vector_store):
        """Test getting course count from empty store"""
        count = vector_store.get_course_count()
        assert count == 0
    
    def test_get_course_count_populated(self, populated_vector_store):
        """Test getting course count from populated store"""
        count = populated_vector_store.get_course_count()
        assert count == 1
    
    def test_clear_all_data(self, populated_vector_store):
        """Test clearing all data from vector store"""
        # Verify we have data first
        assert populated_vector_store.get_course_count() == 1
        
        # Clear the data
        populated_vector_store.clear_all_data()
        
        # Verify data is cleared
        assert populated_vector_store.get_course_count() == 0
        results = populated_vector_store.search("Python")
        assert results.is_empty()
    
    def test_get_all_courses_metadata(self, populated_vector_store):
        """Test retrieving all courses metadata"""
        metadata = populated_vector_store.get_all_courses_metadata()
        
        assert len(metadata) == 1
        course_meta = metadata[0]
        assert course_meta["title"] == "Python Basics"
        assert course_meta["instructor"] == "John Doe"
        assert "lessons" in course_meta
        assert len(course_meta["lessons"]) == 2
    
    def test_get_course_link(self, populated_vector_store):
        """Test retrieving course link"""
        link = populated_vector_store.get_course_link("Python Basics")
        assert link == "http://example.com/course"
        
        # Test with non-existent course
        link = populated_vector_store.get_course_link("Non-existent")
        assert link is None
    
    def test_get_lesson_link(self, populated_vector_store):
        """Test retrieving lesson link"""
        link = populated_vector_store.get_lesson_link("Python Basics", 1)
        assert link == "http://example.com/lesson1"
        
        link = populated_vector_store.get_lesson_link("Python Basics", 2)
        assert link == "http://example.com/lesson2"
        
        # Test with non-existent lesson
        link = populated_vector_store.get_lesson_link("Python Basics", 99)
        assert link is None
    
    @patch('vector_store.chromadb.PersistentClient')
    def test_chroma_connection_error(self, mock_client, test_config):
        """Test handling of ChromaDB connection errors"""
        mock_client.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL)
    
    def test_search_with_exception(self, populated_vector_store):
        """Test search method when ChromaDB raises exception"""
        # Mock the collection to raise an exception
        populated_vector_store.course_content.query = Mock(side_effect=Exception("Query failed"))
        
        results = populated_vector_store.search("test")
        
        assert results.error is not None
        assert "Search error" in results.error


class TestSearchResults:
    """Test suite for SearchResults data class"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'value1'}, {'key': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Test error")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error"
    
    def test_is_empty_true(self):
        """Test is_empty method with empty results"""
        results = SearchResults([], [], [])
        assert results.is_empty() is True
    
    def test_is_empty_false(self):
        """Test is_empty method with results"""
        results = SearchResults(['doc1'], [{'key': 'value'}], [0.1])
        assert results.is_empty() is False