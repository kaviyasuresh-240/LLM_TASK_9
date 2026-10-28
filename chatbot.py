import ollama
import pandas as pd
from datetime import datetime
import os

# File paths
CHAT_HISTORY_FILE = "dataset/chat_history.csv"
RESPONSES_FILE = "dataset/responses.csv"
LOG_FILE = "dataset/chatbot_logs.txt"

# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# System prompt for Healthcare chatbot
conversation_history = [
    {
        "role": "system",
        "content": """
        You are a Healthcare Assistant Chatbot.

        Your responsibilities:
        - Answer healthcare-related questions.
        - Explain concepts clearly.
        - Provide examples where relevant.
        - Give educational information only.
        - Do not provide medical diagnosis.
        - Encourage users to consult healthcare professionals for serious concerns.
        """
    }
]

print("=" * 60)
print("HEALTHCARE CHATBOT USING OLLAMA + GEMMA")
print("Type 'exit' to end the conversation.")
print("=" * 60)

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("\nChatbot terminated successfully.")
        break

    # Add user message to conversation memory
    conversation_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:
        response = ollama.chat(
            model="gemma",
            messages=conversation_history
        )

        bot_response = response["message"]["content"]

        print("\nBot:", bot_response)

        # Store assistant response in memory
        conversation_history.append(
            {
                "role": "assistant",
                "content": bot_response
            }
        )

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to chat_history.csv
        chat_df = pd.DataFrame([
            {
                "Timestamp": timestamp,
                "User": user_input,
                "Bot": bot_response
            }
        ])

        if os.path.exists(CHAT_HISTORY_FILE):
            chat_df.to_csv(
                CHAT_HISTORY_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            chat_df.to_csv(
                CHAT_HISTORY_FILE,
                index=False
            )

        # Save to responses.csv
        response_df = pd.DataFrame([
            {
                "Question": user_input,
                "Response": bot_response
            }
        ])

        if os.path.exists(RESPONSES_FILE):
            response_df.to_csv(
                RESPONSES_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            response_df.to_csv(
                RESPONSES_FILE,
                index=False
            )

        # Save to chatbot_logs.txt
        with open(LOG_FILE, "a", encoding="utf-8") as file:
            file.write(f"\n[{timestamp}]\n")
            file.write(f"USER: {user_input}\n")
            file.write(f"BOT: {bot_response}\n")
            file.write("-" * 60 + "\n")

    except Exception as e:
        print("\nError:", e)

print("\nConversation history saved successfully.")