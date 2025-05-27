import pandas as pd
import ast
from sklearn.utils import shuffle

# Load CSV into DataFrame
df = pd.read_csv("your_data.csv")

# Ensure 'conversation_id' exists; if not, generate it
if "conversation_id" not in df.columns:
    df["conversation_id"] = (df.index // 5)  # Example: Group every 5 messages as one session

# Group by conversation
grouped = df.groupby("conversation_id", sort=False)

# Convert each conversation group into a single text chunk
conversations = []
for _, group in grouped:
    conversation_text = " ".join(group["text_data"])  # Adjust column name as needed
    conversations.append(conversation_text)

# Shuffle conversations for randomness
conversations = shuffle(conversations, random_state=42)

# Split by conversation count (not row count)
split_ratio = 0.9
split_index = int(len(conversations) * split_ratio)

train_data = conversations[:split_index]  # 90% train
test_data = conversations[split_index:]  # 10% test

print(f"Train dataset size: {len(train_data)} conversations")
print(f"Validation dataset size: {len(test_data)} conversations")

# Tokenize and convert into dataset format
train_dataset = tokenizer(train_data, truncation=True, padding="max_length", max_length=128)
test_dataset = tokenizer(test_data, truncation=True, padding="max_length", max_length=128)

train_dataset["labels"] = train_dataset["input_ids"].copy()
test_dataset["labels"] = test_dataset["input_ids"].copy()
