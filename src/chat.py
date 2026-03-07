from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN

class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.system_prompt = (
            "You are an MIT Course Catalog assistant helping students choose classes. "
            "Ask concise follow-up questions to clarify constraints: major/course, class year, "
            "requirements (CI-H, HASS, REST, GIRs), schedule preferences, and interests. "
            "Recommend a short list of relevant classes and explain why each fits the student's constraints. "
            "When useful, include class number, title, likely requirement fit, and common prerequisites. "
            "If schedule/instructor details might vary by term, clearly say they should verify in the live catalog "
            "at student.mit.edu/catalog before enrolling. "
            "Be practical, organized, and concise."
            "Do not invent classes. Prioritize the provided context; if context is limited, say so clearly."
        )
        
    def format_prompt(self, user_input, history=None):
        """
        TODO: Implement this method to format the user's input into a proper prompt.
        
        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question

        Returns:
            str: A formatted prompt ready for the model
        
        Example prompt format:
            "You are a helpful assistant that specializes in...
             User: {user_input}
             Assistant:"
        """
        history = history or []
        history_lines = []
        for user_msg, assistant_msg in history:
            history_lines.append(f"User: {user_msg}")
            history_lines.append(f"Assistant: {assistant_msg}")

        history_block = "\n".join(history_lines).strip()
        if history_block:
            history_block += "\n"

        prompt = (
            f"System: {self.system_prompt}\n\n"
            "Conversation so far:\n"
            f"{history_block}"
            f"User: {user_input}\n"
            "Assistant:"
        )
        return prompt
        
    def get_response(self, user_input, history=None):
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """
        history = history or []

        # Try conversational API first.
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": user_input})

            completion = self.client.chat_completion(
                messages=messages,
                max_tokens=350,
                temperature=0.3,
            )
            content = completion.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            pass

        # Fallback to plain text generation for compatible models/providers -> This one didn't work with deepseek.
        prompt = self.format_prompt(user_input, history=history)
        try:
            raw_response = self.client.text_generation(
                prompt,
                max_new_tokens=250,
                return_full_text=False,
            )
            if raw_response and str(raw_response).strip():
                return str(raw_response).strip()
        except Exception:
            pass

        return "I am having trouble generating a response right now. Please try again in a few seconds."
