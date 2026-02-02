import os
import sys
import argparse
from typing import Optional, List, Dict, Any, Tuple


try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library not installed")
    print("Please run: pip install openai")
    sys.exit(1)


class OpenAIChat:
    """OpenAI Chat Class"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL, if None then use default value
        """
        self.api_key = api_key
        self.base_url = base_url
        
        # Create client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        try:
            self.client = OpenAI(**client_kwargs)
            print(f"OpenAI client initialized successfully")
            if self.base_url:
                print(f"   Using custom baseurl: {self.base_url}")
            else:
                print(f"   Using default baseurl")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            sys.exit(1)
    
    def chat(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> Optional[str]:
        """
        Send chat messages and get response
        
        Args:
            messages: Message list, format: [{"role": "user", "content": "..."}]
            model: Model name to use, default is gpt-3.5-turbo
            
        Returns:
            AI response content, returns None if error occurs
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Chat request failed: {e}")
            return None
    
    def start_conversation(self, model: str = "gpt-3.5-turbo", system_prompt: Optional[str] = None):
        """
        Start interactive conversation
        
        Args:
            model: Model name to use
            system_prompt: System prompt, used to set AI's role and behavior
        """
        print("\n" + "="*60)
        print("OpenAI Chat Tool")
        print("="*60)
        print(f"Using model: {model}")
        if system_prompt:
            print(f"System prompt: {system_prompt}")
        print("\nTips:")
        print("  - Type 'quit' or 'exit' to exit conversation")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'model <model_name>' to switch model")
        print("="*60 + "\n")
        
        # Initialize message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        conversation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle special commands
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    conversation_count = 0
                    print("Conversation history cleared\n")
                    continue
                
                if user_input.lower().startswith('model '):
                    new_model = user_input[6:].strip()
                    model = new_model
                    print(f"Switched to model: {model}\n")
                    continue
                
                # Add user message
                messages.append({"role": "user", "content": user_input})
                conversation_count += 1
                
                # Show thinking prompt
                print("AI is thinking...", end="", flush=True)
                
                # Get AI response
                response = self.chat(messages, model=model)
                
                # Clear thinking prompt
                print("\r" + " " * 30 + "\r", end="")
                
                if response:
                    # Add AI response to message list
                    messages.append({"role": "assistant", "content": response})
                    print(f"AI: {response}\n")
                else:
                    # If error, remove last user message
                    messages.pop()
                    conversation_count -= 1
                    print("Failed to get response, please try again\n")
                    
            except KeyboardInterrupt:
                print("\n\nConversation interrupted, goodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError occurred: {e}\n")


def get_config_from_env() -> Tuple[Optional[str], Optional[str]]:
    """
    Get configuration from environment variables
    
    Returns:
        (api_key, base_url) tuple
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    return api_key, base_url


def get_config_from_input() -> Tuple[str, Optional[str]]:
    """
    Get configuration through user input
    
    Returns:
        (api_key, base_url) tuple
    """
    print("="*60)
    print("Configure OpenAI API")
    print("="*60)
    
    api_key = input("Please enter OpenAI API Key (required): ").strip()
    if not api_key:
        print("API Key cannot be empty")
        sys.exit(1)
    
    base_url = input("Please enter Base URL (optional, press Enter to use default): ").strip()
    if not base_url:
        base_url = None
    
    return api_key, base_url


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="OpenAI Chat Tool - Supports custom baseurl and API key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use environment variables
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://api.openai.com/v1"
  python talk2ai.py

  # Use command line arguments
  python talk2ai.py --key "sk-..." --baseurl "https://api.openai.com/v1"

  # Interactive input
  python talk2ai.py

  # Specify model
  python talk2ai.py --model "gpt-4"

  # Set system prompt
  python talk2ai.py --system "You are a professional stock analyst"
        """
    )
    
    parser.add_argument(
        "--key",
        "--api-key",
        default=None,
        dest="api_key",
        help="OpenAI API Key (required if not set in environment variable OPENAI_API_KEY)"
    )
    
    parser.add_argument(
        "--baseurl",
        "--base-url",
        default=None,
        dest="base_url",
        help="OpenAI API Base URL (default: https://api.openai.com/v1 if not set)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="Model name to use (default: gpt-5-nano)"
    )
    
    parser.add_argument(
        "--system",
        "--system-prompt",
        dest="system_prompt",
        help="System prompt"
    )
    
    args = parser.parse_args()
    
    # Get configuration: priority is command line arguments > environment variables > user input
    api_key = args.api_key
    base_url = args.base_url
    
    if not api_key:
        api_key, env_base_url = get_config_from_env()
        if not base_url:
            base_url = env_base_url or "https://api.openai.com/v1"
    
    if not api_key:
        api_key, input_base_url = get_config_from_input()
        if not base_url:
            base_url = input_base_url or "https://api.openai.com/v1"
    
    # If base_url is still empty, use default value
    if not base_url:
        base_url = "https://api.openai.com/v1"
    
    # Create chat instance
    try:
        chat = OpenAIChat(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)
    
    # Start conversation
    chat.start_conversation(
        model=args.model,
        system_prompt=args.system_prompt
    )


if __name__ == "__main__":
    main()

