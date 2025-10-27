⸻
“Add Language Encoder to Ada (Phase 3 Conversational Fix)”

Integrate a text-embedding encoder into Ada’s architecture so she can process
and understand user input semantically rather than as zero vectors.

Goal:
Add a lightweight SentenceTransformer-based encoder module,
connect it to the AdaCore forward path, and update the CLI and trainer
to use real language embeddings for input instead of random or null tensors.
⸻

🧩 Core Objectives 1. Add a new module /neural/encoder.py implementing LanguageEncoder. 2. Modify /interfaces/cli.py to encode user text before passing to AdaCore. 3. Update /neural/policy_network.py forward method to accept embeddings directly. 4. Ensure AdaCore checkpoints remain compatible. 5. Verify confidence values now vary dynamically (non-zero).

⸻

🧠 New Module

/neural/encoder.py

from sentence_transformers import SentenceTransformer
import torch

class LanguageEncoder:
def **init**(self, model_name="all-MiniLM-L6-v2"):
self.model = SentenceTransformer(model_name)
def encode(self, text: str) -> torch.Tensor:
vec = self.model.encode([text], normalize_embeddings=True)
return torch.tensor(vec, dtype=torch.float32)

⸻

🧩 Update AdaCore Forward Path

/neural/policy_network.py

import torch
import torch.nn as nn

class AdaCore(nn.Module):
def **init**(self, input_dim=384, hidden_dim=256, output_dim=512):
super().**init**()
self.fc1 = nn.Linear(input_dim, hidden_dim)
self.relu = nn.ReLU()
self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

(Input size adjusted to 384 to match MiniLM embeddings.)

⸻

🧩 Update CLI Interface

/interfaces/cli.py

from neural.encoder import LanguageEncoder
from neural.policy_network import AdaCore
import torch

encoder = LanguageEncoder()
model = AdaCore()
model.load_state_dict(torch.load("storage/checkpoints/ada_core.pt", map_location="cpu"))
model.eval()

while True:
user_input = input("You: ")
if user_input.lower() in ["quit", "exit"]:
print("Ada: Until next time.")
break

    x = encoder.encode(user_input)
    with torch.no_grad():
        y = model(x)
    confidence = float(torch.softmax(y, dim=-1).max())
    print(f"Ada: [generated placeholder] (confidence: {confidence:.2f})")

⸻

🧩 Trainer Update

/neural/trainer.py

Add before model training:

from neural.encoder import LanguageEncoder
encoder = LanguageEncoder()

Then replace any random input tensors with:

x = encoder.encode(text_sample)

⸻

🧩 Dependencies

In requirements.txt:

sentence-transformers
torch

⸻

📦 Deliverables 1. New /neural/encoder.py 2. Updated /interfaces/cli.py and /neural/policy_network.py 3. Adjusted input dimensions in trainer 4. Verified output confidence now > 0.00 5. README update:
“Phase 3 Conversational Fix — Ada now embeds user input using SentenceTransformers (MiniLM-L6-v2).”

⸻

🧩 Expected Result

You: hi ada
Ada: Hello, Taahirah! How’s your day going? (confidence: 0.47)
You: I’m good
Ada: That’s great to hear. (confidence: 0.59)
⸻
