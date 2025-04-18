# Building a Board Game Rule Assistant with RAG and Generative AI
This repository is meant to be a blogpost for Gen AI Intensive Course Capstone 2025Q1 to explain a RAG application which helps people understand the rules of any board game.

## Problem: Understanding Board Game Rules is Painful
Ever been excited to play a new board game, only to get stuck in 20 pages of rules? It slows down the fun, creates confusion, and causes arguments about interpretations. The problem is that board game rulebooks are long, complex, and full of exceptions. Players often just want to ask a simple question like:
- “How many resources do I need to trade at a port in Catan?”
- “How do I get out of this trap?”
- “Is it allowed to attack the king without these tools?”

But players have no easy way to find that out quickly.

## Solution: Gen AI + RAG = Real-Time Game Rule Q&A
By combining Generative AI with Retrieval-Augmented Generation (RAG), we can build a smart assistant that:
- Understands board game rulebooks
- Accepts natural-language questions
- Retrieves relevant rules
- Generates friendly, clear answers

No more flipping through manuals — just ask your question.

## Framework: A RAG Pipeline with Gemini + Chroma
Here’s how the system works:
- Load and segment rule documents
- Convert each chunk into embeddings (vectors)
- Store in a vector database (ChromaDB)
- On user query: retrieve relevant chunks, then generate a response using Gemini

## Code Snippets
### Define the documents (game rules)
```
GAME_RULES = {...}
```
### Create a custom Gemini embedding function
```
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True
    
    @retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError))
    def __call__(self, input: Documents) -> Embeddings:
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]
```

### Add documents to ChromaDB
```
embed_fn = GeminiEmbeddingFunction()
chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(
    name="boardgames_db", 
    embedding_function=embed_fn)

db.add(
    documents=list(GAME_RULES.values()),
    ids=list(GAME_RULES.keys()),
    metadatas=[{"game": name} for name in GAME_RULES.keys()]
)
```

### Answer user questions via retrieval + generation
```
def ask_about_game(question, game_name=None):
    try:
        # Query with optional game filtering
        filters = {"game": game_name} if game_name else None
        results = db.query(
            query_texts=[question],
            n_results=3,  # Get top 3 most relevant passages
            where=filters,
        )
        
        passage = results["documents"][0][0]

        # Few-Shot Prompting
        prompt = f"""Act as a board game expert. Use these rules to explain to the player with details:
        Examples:
        Q: How do ports work in Catan?  
        A: Ports allow 2:1 or 3:1 resource trades. 
        Now answer:  
        QUESTION: {question}
        Answer ONLY using these rules:{passage}.
        If unsure, say "I don't have rules for that"."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt)
        return response.text
    
    except Exception as e:
        return f"Error: {str(e)}"
```

## Limitations
While this RAG-based assistant is powerful and surprisingly effective, it does come with a few limitations. One challenge lies in how documents are chunked before being embedded. If the chunks are too small, the model might miss important context; too large, and the retrieval step can get noisy or irrelevant. Balancing this trade-off is more of an art than a science. Another limitation is the system's ability to handle complex, multi-step logic — which is common in many advanced board games. Scenarios that require reasoning across several interconnected rules or conditional outcomes are still tough for current LLMs to navigate reliably. Additionally, this implementation is currently built for plain-text documents. Many game manuals include diagrams, tables, or visual setups that are crucial for understanding gameplay, but aren’t accessible to a text-based system. Extending support for those richer content types remains a challenge. Finally, there’s the practical matter of cost. Using APIs for embedding and generation — especially at scale — can add up. While fine for small-scale or personal projects, larger deployments would need to monitor usage and optimize costs.

## Future Possibilities
Despite the limitations, the future of this project is exciting, with lots of room to grow and improve.
- One natural next step is to expand the assistant to support multiple games. By simply indexing different rulebooks into the same system, the assistant could become a one-stop hub for answering rules across a whole library of board games.
- Integrating voice input would make the tool even more seamless during actual gameplay. Imagine asking, “Can I build a road here?” and getting an instant spoken answer without ever needing to touch a phone or pause the game.
- There’s also room to support more natural, multi-turn conversations. For example, players could ask follow-up questions or refine their query based on the last response — just like chatting with a real human rule master.
- Lastly, a truly next-gen version of this assistant could maintain a model of the game state: keeping track of player resources, board positions, or actions taken — and then validating whether a move is legal or not in real time.

