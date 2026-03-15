"""
Gradio Web Interface for MIT Course Catalog Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from src.chat import Chatbot

def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = Chatbot()
    
    def chat(message, history):
        """
        Generate a response for the current message in a Gradio chat interface.
        
        This function is called by Gradio's ChatInterface every time a user sends a message.
        You only need to generate and return the assistant's response - Gradio handles the
        chat display and history management automatically.

        Args:
            message (str): The current message from the user
            history (list): List of previous message pairs, where each pair is
                           [user_message, assistant_message]
                           Example:
                           [
                               ["What schools offer Spanish?", "The Hernandez School..."],
                               ["Where is it located?", "The Hernandez School is in Roxbury..."]
                           ]

        Returns:
            str: The assistant's response to the current message.


        Note:
            - Gradio automatically:
                - Displays the user's message
                - Displays your returned response
                - Updates the chat history
                - Maintains the chat interface
            - You only need to:
                - Generate an appropriate response to the current message
                - Return that response as a string
        """
        if not message or not message.strip():
            return "Share your course planning goals and constraints, and I can suggest classes."

        return chatbot.get_response(message, history=history)

    
    
    # Create Gradio interface. Customize the interface however you'd like!
    demo = gr.ChatInterface(
        chat,
        title="MIT Course Catalog Navigator",
        description=(
            "I can help you find MIT classes that match your requirements and interests. "
            "Tell me your course number, requirements (CI-H/HASS/REST), and schedule preferences. "
            "Since this app uses the free tier, occasional 503 errors can happen; try again in a few seconds."
        ),
        examples=[
            "I am a 6-3 junior and need a CI-H. I prefer afternoon classes and I am interested in AI ethics.",
            "I need one HASS and one REST next term with no Friday classes. What options should I consider?",
            "Can you suggest intro-level ML-adjacent classes that do not require heavy math prerequisites?"
        ]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
