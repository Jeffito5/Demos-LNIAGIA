# Robot Manel - LNIAGIA Project Documentation

## Table of Contents
1. [Full Action Plan](#full-action-plan)
2. [Complete Explanation of the Naive ChatBot](#complete-explanation-of-the-naive-chatbot)
3. [How to Test/Adapt for Your Project](#how-to-testadapt-this-for-your-project)

---

## Full Action Plan

### Phase 1 — Run and Understand the Naive ChatBot (Current Priority)

**Step 1: Set up the environment**
- Install dependencies: `scikit-learn`, `pandas`, `numpy`, `nltk`, `chatterbot`, `spacy`
- Download spaCy model: `python -m spacy download en_core_web_sm`
- Fix the known ChatterBot/spaCy bug (line 13 of `site-packages/chatterbot/tagging.py`)

**Step 2: Get the data**
- Ensure `data/dialogues.tsv`, `data/tagged_posts.tsv`, and `data/word_embeddings.tsv` exist inside `Demos/Naive ChatBot/data/`
- These are StackOverflow + movie dialogue datasets + StarSpace embeddings provided by the teacher

**Step 3: Run the notebook end-to-end**
- Run `assistant-bot.ipynb` cell by cell
- This generates: `intent_recognizer.pkl`, `tag_classifier.pkl`, `tfidf_vectorizer.pkl`, `thread_embeddings_by_tags/`
- Test the dialogue loop at the end — verify it classifies intents and returns answers

**Step 4: Understand every component** (see complete explanation section below)

### Phase 2 — Adapt the Naive ChatBot to Your Project Domain

**Step 5: Replace the domain**
- Instead of StackOverflow questions -> **robot commands** (e.g., "pick up the screw", "move to station 2")
- Instead of movie dialogues -> **operator chitchat** (e.g., "how are you", "what can you do")
- Use the new datasets in `robot manel/data/`: `robot_commands.tsv` and `operator_chitchat.tsv`

**Step 6: Replace tag classification**
- Instead of programming languages -> **action categories** (e.g., PICK_PLACE, NAVIGATE, ASSEMBLE, TOOL_ASSIST, STATUS, MEASURE, SAND, CUT, DRILL, GLUE, CLAMP)
- Train the same OneVsRest LogisticRegression on your action categories

**Step 7: Replace thread ranking with action lookup**
- Instead of finding the best StackOverflow thread -> **map to a specific robotic action plan**
- Use the action plan database in `action_plans.json`

**Step 8: Replace ChatterBot with something better**
- ChatterBot is outdated — the adapted version uses a simple rule-based fallback for now
- In Phase 3, replace with an LLM

### Phase 3 — Evolve Beyond "Naive" (Final Project)

**Step 9: Upgrade to Transformers**
- Replace TF-IDF + LogisticRegression with a fine-tuned BERT/DistilBERT for intent + action classification
- This aligns with lectures 07 (Transformers) and 08 (LLMs) from the course

**Step 10: Add multi-turn dialogue**
- Implement conversation state tracking (the naive bot is stateless — each question is independent)
- This aligns with lecture 06 (Conversational Systems)

**Step 11: Integrate LLM for action plan generation**
- Use an LLM to convert natural language instructions into structured action plans
- This is the core of the LNIAGIA proposal

**Step 12: Add MCP-based memory**
- Implement evolutionary memory that stores operator corrections and adapts over time
- Stores history of corrections, adapts to each operator's style
- Builds a progressive knowledge base for the company

**Step 13: Add Speech-to-Text**
- Integrate Whisper or similar ASR as the input pipeline
- This connects the NLP system to the robot's microphone

---

## Complete Explanation of the Naive ChatBot

The Naive ChatBot is a **goal-oriented dialogue system** with a chitchat fallback. It has 6 main components that work together in a pipeline.

### Architecture Overview

```
User Input
    |
    v
+---------------------+
|   text_prepare()    |  <- Lowercase, remove symbols, remove stopwords
+---------+-----------+
          |
          v
+---------------------+
|  TF-IDF Vectorizer  |  <- Convert text -> sparse numerical vector (ngram 1-2)
+---------+-----------+
          |
          v
+---------------------+
|  Intent Recognizer  |  <- Binary LogisticRegression: "dialogue" or "stackoverflow"?
+---------+-----------+
          |
    +-----+------+
    v            v
DIALOGUE    STACKOVERFLOW
    |            |
    v            v
+--------+  +--------------+
|Chatter |  |Tag Classifier|  <- Which programming language? (OneVsRest LR)
|  Bot   |  +------+-------+
+--------+         |
                   v
              +--------------+
              |Thread Ranker |  <- Find best SO thread using word embeddings
              +--------------+
```

### Component 1: Text Preprocessing (utils.py:19-30)

```python
def text_prepare(text):
```
- Converts to lowercase
- Replaces `/(){}[]|@,;` with spaces
- Removes anything that isn't `0-9 a-z # + _`
- Removes English stopwords ("the", "is", "at", etc.)
- Example: `"How to write a LOOP in Python?"` -> `"write loop python"`

**Why?** Reduces noise so the classifier focuses on meaningful words. Without this, "How to write a loop" and "Writing loops" would look very different to the model.

### Component 2: TF-IDF Vectorizer (assistant-bot.ipynb cell 12)

```python
TfidfVectorizer(min_df=5, max_df=9000, ngram_range=(1,2), token_pattern=r'(\S+)')
```

**What is TF-IDF?**
- **TF (Term Frequency)**: How often a word appears in this document
- **IDF (Inverse Document Frequency)**: How rare the word is across all documents
- **TF-IDF = TF x IDF**: Words that are frequent in THIS text but rare overall get high scores

**Parameters:**
- `min_df=5`: Ignore words appearing in fewer than 5 documents (removes typos/noise)
- `max_df=9000`: Ignore words appearing in more than 9000 documents (too common)
- `ngram_range=(1,2)`: Consider single words AND pairs ("python loop", not just "python" + "loop")

**Output**: A sparse matrix where each row is a document and each column is a TF-IDF score for a specific word/bigram. This is what the classifiers receive as input.

The vectorizer is trained on 360,000 texts (200k dialogues + 200k StackOverflow) and pickled to `tfidf_vectorizer.pkl` for reuse at runtime.

### Component 3: Intent Recognizer (assistant-bot.ipynb cells 27-28)

```python
LogisticRegression(penalty='l2', C=10, random_state=0)
```

**Task**: Binary classification - is the user asking a programming question (`stackoverflow`) or just chatting (`dialogue`)?

**Training data**:
- 200,000 movie dialogue lines (labeled `dialogue`)
- 200,000 StackOverflow titles (labeled `stackoverflow`)
- Split 90/10 -> 360k train, 40k test

**Result**: 99.06% accuracy - essentially perfect. This works because programming questions use very different vocabulary from casual conversation.

**How LogisticRegression works here**: It learns a weight for each TF-IDF feature. Words like "python", "loop", "function", "error" get high positive weights (-> stackoverflow). Words like "hello", "love", "dinner" get high negative weights (-> dialogue). At prediction time, it sums weighted features and applies a sigmoid to get a probability.

### Component 4: Tag Classifier (assistant-bot.ipynb cells 39-40)

```python
OneVsRestClassifier(LogisticRegression(penalty='l2', C=5))
```

**Task**: Given that we know it's a programming question, which language is it about? (10 classes: Python, Java, C#, C++, JavaScript, PHP, Ruby, R, Swift, VB)

**Training data**: 200,000 StackOverflow titles with their language tags, split 80/20

**Result**: 79.15% accuracy - lower because distinguishing "Java vs C#" is much harder than "dialogue vs code"

**OneVsRest**: Trains 10 separate binary classifiers (one per language). Each one learns "is this Python or not?", "is this Java or not?", etc. The language with the highest confidence wins.

### Component 5: Thread Ranker (dialogue_manager.py:12-35)

**Task**: Once we know the language, find the most relevant StackOverflow thread.

**How it works**:
1. Pre-computed: Every StackOverflow title is converted to a dense vector using StarSpace word embeddings (average of word vectors)
2. These are stored by tag in `thread_embeddings_by_tags/{language}.pkl`
3. At runtime: The user's question is also converted to a vector
4. Find the thread whose vector is closest (Euclidean distance) to the question vector
5. Return the StackOverflow URL

**Why embeddings instead of TF-IDF?** TF-IDF is sparse and keyword-based - "how to iterate" and "looping through" would have zero overlap. Embeddings capture semantic similarity - both would be close in vector space.

### Component 6: Dialogue Manager (dialogue_manager.py:38-100)

This is the **orchestrator** that ties everything together:

```python
def generate_answer(self, question):
    prepared = text_prepare(question)        # Clean the text
    features = tfidf_vectorizer.transform()  # Vectorize
    intent = intent_recognizer.predict()     # Dialogue or SO?

    if intent == 'dialogue':
        return chitchat_bot.get_response()   # ChatterBot handles it
    else:
        tag = tag_classifier.predict()       # Which language?
        thread = thread_ranker.get_best()    # Best SO thread
        return ANSWER_TEMPLATE % (tag, thread)
```

**ChatterBot**: A pre-trained retrieval-based chatbot. It was trained on the English corpus in `chatterbot_corpus/data/english/` (greetings, conversations, AI topics, etc.). It finds the closest match to the user's input from its training data and returns the paired response.

---

## How to Test/Adapt This for Your Project

The key insight: **the notebook's architecture maps directly to your robotic system**. Here's the translation:

| Naive ChatBot | Your Robot System |
|---|---|
| "dialogue" vs "stackoverflow" | "chitchat" vs "robot_command" |
| Programming language tag | Action category (CUT, DRILL, SAND, ASSEMBLE...) |
| Best StackOverflow thread | Best action plan/sequence |
| ChatterBot response | Conversational response ("I can help with X") |
| Movie dialogue data | Operator small talk corpus |
| StackOverflow titles | Robot command phrases |

### Concrete Testing Plan

**Test 1: Run the original notebook as-is**
- Make sure you have the data files (`dialogues.tsv`, `tagged_posts.tsv`, `word_embeddings.tsv`)
- Run every cell, verify you get ~99% intent accuracy and ~79% tag accuracy
- Chat with it - test edge cases (ambiguous questions, short inputs)

**Test 2: Create a minimal robot-command version**
- Use the provided `robot_commands.tsv` in `robot manel/data/`
- Use the provided `operator_chitchat.tsv` in `robot manel/data/`

**Test 3: Retrain the intent recognizer** on the new data (robot commands vs chitchat)

**Test 4: Retrain the tag classifier** on the action categories

**Test 5: Replace thread ranking** with action plan lookup using `action_plans.json`

**Test 6: Verify the "two phrases = same task" requirement**
- Input "pick up the screw" -> should classify as PICK_PLACE
- Input "grab that component" -> should also classify as PICK_PLACE
- Input "cut this board in half" -> should classify as CUT
- Input "saw through the plank" -> should also classify as CUT
- This proves the NLP system understands intent regardless of phrasing

### Files in robot manel/

- `PLAN.md` - This documentation file
- `data/robot_commands.tsv` - Robot command phrases labeled by action category (carpentry-focused)
- `data/operator_chitchat.tsv` - Operator small talk phrases
- `action_plans.json` - Action category -> execution steps mapping
- `utils.py` - Adapted utility functions for the robot domain
- `dialogue_manager.py` - Adapted DialogueManager for robot commands
- `robot_assistant.ipynb` - Notebook to train and test the robot command system
