import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_client:
            yield mock_client.return_value
    
    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator instance with mocked client"""
        return AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20241022")
    
    def test_init(self, ai_generator, mock_anthropic_client):
        """Test AIGenerator initialization"""
        assert ai_generator.model == "claude-3-5-sonnet-20241022"
        assert ai_generator.base_params["model"] == "claude-3-5-sonnet-20241022"
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800
    
    def test_generate_response_simple(self, ai_generator, mock_anthropic_client):
        """Test simple response generation without tools"""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = ai_generator.generate_response("What is Python?")
        
        assert result == "Test response"
        
        # Verify API call parameters
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert "tools" not in call_args[1]
    
    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        history = "User: Hello\nAssistant: Hi there!"
        result = ai_generator.generate_response("How are you?", conversation_history=history)
        
        assert result == "Response with history"
        
        # Check that system prompt includes history
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content
    
    def test_generate_response_with_tools_no_tool_use(self, ai_generator, mock_anthropic_client):
        """Test response generation with tools available but not used"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = [{"name": "search_tool", "description": "Search for content"}]
        tool_manager = Mock()
        
        result = ai_generator.generate_response(
            "What is 2+2?", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        assert result == "Direct response"
        
        # Verify tools were included in API call
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client):
        """Test response generation when AI decides to use tools"""
        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "Python basics"}
        mock_initial_response.content = [mock_tool_block]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Based on search results: Python is great!")]
        
        # Set up mock client to return different responses
        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Python is a programming language"
        
        tools = [{"name": "search_course_content", "description": "Search content"}]
        
        result = ai_generator.generate_response(
            "Tell me about Python", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        assert result == "Based on search results: Python is great!"
        
        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="Python basics"
        )
        
        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self, ai_generator, mock_anthropic_client):
        """Test _handle_tool_execution method with single tool"""
        # Create mock initial response with tool use
        mock_initial_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_tool"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}
        mock_initial_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }
        
        result = ai_generator._handle_tool_execution(
            mock_initial_response, 
            base_params, 
            tool_manager
        )
        
        assert result == "Final answer"
        tool_manager.execute_tool.assert_called_once_with("search_tool", query="test")
        
        # Verify the message flow
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        
        # Should have: original user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
    
    def test_handle_tool_execution_multiple_tools(self, ai_generator, mock_anthropic_client):
        """Test _handle_tool_execution method with multiple tools"""
        # Create mock initial response with multiple tool uses
        mock_initial_response = Mock()
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "tool1"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"query": "test1"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "tool2"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"query": "test2"}
        
        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt"
        }
        
        result = ai_generator._handle_tool_execution(
            mock_initial_response, 
            base_params, 
            tool_manager
        )
        
        assert result == "Final answer"
        assert tool_manager.execute_tool.call_count == 2
        
        # Check tool results were properly formatted
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        tool_results = messages[2]["content"]
        
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Result 1"
        assert tool_results[1]["tool_use_id"] == "tool_456"
        assert tool_results[1]["content"] == "Result 2"
    
    def test_handle_tool_execution_no_tool_blocks(self, ai_generator, mock_anthropic_client):
        """Test _handle_tool_execution with response containing no tool blocks"""
        mock_initial_response = Mock()
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_initial_response.content = [mock_text_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        tool_manager = Mock()
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "system"
        }
        
        result = ai_generator._handle_tool_execution(
            mock_initial_response, 
            base_params, 
            tool_manager
        )
        
        assert result == "Final answer"
        tool_manager.execute_tool.assert_not_called()
        
        # Should still make final API call but with no tool results
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2  # Only user + assistant messages, no tool results
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_anthropic_api_error(self, mock_anthropic):
        """Test handling of Anthropic API errors"""
        mock_client = mock_anthropic.return_value
        mock_client.messages.create.side_effect = Exception("API Error")
        
        ai_gen = AIGenerator("test_key", "test_model")
        
        with pytest.raises(Exception, match="API Error"):
            ai_gen.generate_response("test query")
    
    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains expected instructions"""
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        # Check for key instructions
        assert "search tool" in system_prompt.lower()
        assert "course content" in system_prompt.lower()
        assert "one search per query maximum" in system_prompt.lower()
        assert "brief" in system_prompt.lower()
        assert "educational" in system_prompt.lower()