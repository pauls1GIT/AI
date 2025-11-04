**Bolt Attribute Extractor**

Bolt Attribute Extractor is a desktop tool that uses AI (OpenAI GPT-4o and Tavily search) to automatically identify key attributes — such as diameter, head type, and length — from images or technical drawings of bolts.

What it does:

1. Prepares context
Tavily searches the web for relevant reference information (e.g., “how to read technical drawings for bolts”).

2. Reads and analyzes images with GPT-4o
Each image is encoded and sent to OpenAI’s GPT-4o model, which is asked three specific questions:

What is the diameter?

What is the head type?

What is the length?

3. Structured output
The AI returns structured answers in an AnswerBlock format:

AnswerBlock(answer="M10", confidence=0.92, reasoning="Label near head section shows Ø10mm")

4. Display & export
The GUI displays results in a table.
Low-confidence results appear in red.
You can export all results to a .csv file.
