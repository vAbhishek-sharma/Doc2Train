#data_structurer.py
import json
import os
import pdb
import sys
from openai import OpenAI
from config import MODELS
from datetime import datetime

# Read environment variables for role and content
DEFAULT_ROLE = os.getenv('DEFAULT_ROLE', 'system')  # Default to 'system' if not set
DEFAULT_ROLE_CONTENT = os.getenv('DEFAULT_ROLE_CONTENT', "You are an expert in semantic data structuring.")
DEFAULT_USER_CONTENT = os.getenv('DEFAULT_USER_CONTENT', "Convert the following text into structured data: {content}")

def get_api_client(provider, model):
    """Initialize API client based on provider and model"""
    if provider not in MODELS or model not in MODELS[provider]:
        raise ValueError(f"Invalid provider or model: {provider} - {model}")

    api_key = os.getenv('OPENAI_API_KEY') if provider == 'openai' else os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError(f"{provider.upper()} API key is not set in environment variables")

    return OpenAI(api_key=api_key, base_url=MODELS[provider][model]['base_url'])


def process_content(content, provider, model, input_file=None, output_format="json"):
    """Process content through selected AI model and generate follow-up questions"""
    try:
        client = get_api_client(provider, model)
        config = MODELS[provider][model]

        user_content = f"""
            Generate a natural multi-turn conversation between the user and AI with at least 3 follow-up interactions based on the following information:

            {content}

            **Strict Formatting Rules:**
            Every user question and AI response **must be separated by ---** [three dashes]. The conversation must always follow this pattern:

            User: <user message>
            Ai: <AI response>
            ---
            User: <next message>
            Ai: <next response>
            ---

            Ensure there are no missing or extra --- separators. Keep responses **positive, motivational, and insightful.**
            Ensure every interaction follows this format with **no missing separators**.
            """

        # Add follow-up prompt for generating multi-turn conversation
        follow_up_prompt = "\n\nGenerate follow-up questions based on the content above, and provide AI responses for them."

        if output_format == "json":
            user_content += "\n\nRespond strictly in JSON format."
        else:
            user_content += "\n\nProvide a structured textual response."

        print(user_content)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": DEFAULT_ROLE, "content": DEFAULT_ROLE_CONTENT},
                    {"role": "user", "content": user_content + follow_up_prompt}  # Adding the follow-up prompt
                ],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            structured_data = response.choices[0].message.content.strip()
            print(response)

            save_raw_data(content, structured_data, input_file)  # Save raw response before returning

            return structured_data

        except openai.RateLimitError as e:
            error_message = f"Error: Rate limit exceeded. {e}"

        except openai.OpenAIError as e:
            error_message = f"OpenAI API Error: {e}"

        except Exception as e:
            error_message = f"Unexpected Error: {e}"

        print(error_message)
        raise RuntimeError(error_message)  # Raise error instead of exiting

    except Exception as e:
        print(f"Error processing content with {provider} - {model}: {str(e)}")
        return None

def save_raw_data(content, response, input_filename='test'):
    """Save raw API response with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_data_dir = f"{input_filename}_raw_responses"
    os.makedirs(raw_data_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_filename))[0]
    filename = os.path.join(raw_data_dir, f"{base_filename}_response_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("=== User Input ===\n")
        file.write(content + "\n\n")
        file.write("=== AI Response ===\n")
        file.write(response + "\n")

    print(f"Raw response saved to {filename}")


