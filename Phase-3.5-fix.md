⸻

“Ada Phase 3.5 fix: Conversational Memory & Semantic Recall”

Extend Ada’s conversational system with a memory layer that stores, embeds, and retrieves
past dialogue context using the LanguageEncoder from Phase 3.

Goal:
Give Ada the ability to recall relevant past exchanges semantically,
so her replies can reference prior sessions, topics, or emotional tone.


⸻

🧩 Core Objectives
	1.	Add an /memory/episodic_store.py module for storing & searching conversations.
	2.	Store each (user_input, ada_response, reward, embedding) tuple after every exchange.
	3.	Add a semantic retrieval function using cosine similarity between the new input and stored embeddings.
	4.	Inject retrieved context back into AdaCore’s input pipeline before response generation.
	5.	Maintain a small DuckDB or SQLite file for persistence.

⸻

🧠 New Module

/memory/episodic_store.py

import sqlite3
import torch
import numpy as np
from torch.nn.functional import cosine_similarity

class EpisodicStore:
    def __init__(self, db_path="storage/conversations.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            ada_response TEXT,
            reward REAL,
            embedding BLOB
        )
        """)

    def store(self, user_input, ada_response, reward, embedding: torch.Tensor):
        emb = embedding.numpy().tobytes()
        self.conn.execute("INSERT INTO memory (user_input, ada_response, reward, embedding) VALUES (?, ?, ?, ?)",
                          (user_input, ada_response, reward, emb))
        self.conn.commit()

    def retrieve(self, query_embedding: torch.Tensor, top_k=3):
        cursor = self.conn.execute("SELECT user_input, ada_response, embedding FROM memory")
        rows = cursor.fetchall()
        if not rows:
            return []
        sims = []
        for user_input, ada_response, emb_blob in rows:
            emb = torch.tensor(np.frombuffer(emb_blob, dtype=np.float32))
            sim = cosine_similarity(query_embedding, emb.unsqueeze(0))
            sims.append((float(sim), user_input, ada_response))
        sims.sort(reverse=True)
        return sims[:top_k]


⸻

🧩 Integration

/interfaces/cli.py

Add imports:

from neural.encoder import LanguageEncoder
from memory.episodic_store import EpisodicStore

Initialize at the top:

encoder = LanguageEncoder()
memory = EpisodicStore()

Modify the main loop:

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Ada: Until next time.")
        break

    q_emb = encoder.encode(user_input)

    # Retrieve related memories
    recalls = memory.retrieve(q_emb)
    context_text = " ".join([r[1] + " " + r[2] for r in recalls])
    full_input = context_text + " " + user_input if recalls else user_input

    x = encoder.encode(full_input)
    with torch.no_grad():
        y = model(x)
    confidence = float(torch.softmax(y, dim=-1).max())

    response = "[placeholder response]"  # to be generated or mapped later
    print(f"Ada: {response} (confidence: {confidence:.2f})")

    memory.store(user_input, response, 0.0, q_emb)


⸻

🧩 Configuration Update

/config/settings.yaml

memory:
  episodic_enabled: true
  db_path: storage/conversations.db
  recall_top_k: 3


⸻

📦 Deliverables
	1.	/memory/episodic_store.py
	2.	Integration hooks in /interfaces/cli.py
	3.	New database table for persistent conversation history
	4.	Updated config with episodic_enabled flag
	5.	README section: Phase 3.5 – Conversational Memory & Recall

⸻

🧠 Expected Behavior

You: hey ada
Ada: Hello again, Taahirah! (confidence: 0.44)
You: remember what we talked about yesterday?
Ada: You asked about integrating Whisper.cpp — want me to check that again? (confidence: 0.62)


⸻

🧩 Next Step – Phase 4 Memory Consolidation

Once this memory system is working, the following phase evolves Ada’s recollection into long-term persona learning — where she uses stored conversations to adjust tone and personality across sessions.

⸻