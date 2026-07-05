# Langchain-And-Langgraaph--Interview-Questions

LangGraph is a library within the LangChain ecosystem that provides a framework for defining, coordinating, and executing multiple LLM agents (or chains) in a structured and efficient manner.

What Are Vector Embeddings? An Intuitive Explanation
Vector embeddings are numerical representations of words or phrases that capture their meanings and relationships, helping machine learning models understand text more effectively.

What Are Vector Embeddings? 
Vector embeddings are digital fingerprints for words or other pieces of data. Instead of using letters or images, they use numbers that are arranged in a specific structure called a vector, which is like an ordered list of values.
Vector embeddings are a valuable technique for transforming complex data into a format suitable for machine learning algorithms. By converting high-dimensional and categorical data into lower-dimensional, continuous representations, embeddings enhance model performance and computational efficiency while preserving underlying data patterns.


What Is Retrieval Augmented Generation (RAG)?
Large language models (LLMs) like have brought remarkable progress, but they come with limitations: outdated knowledge, hallucinations, and generic responses. RAG solves this by grounding model outputs in Retrieval Augmented Generation (RAG).
RAG connects LLMs to external data sources, letting them retrieve relevant information at query time instead of relying solely on training data
The pipeline has five stages: data collection, chunking, embedding, retrieval, and generation
RAG reduces hallucinations, keeps responses current, and supports domain-specific knowledge without retraining the model
Key challenges include chunking strategy, embedding quality, and data freshness

Hallucinations: LLMs sometimes generate plausible-sounding but factually incorrect information, a problem known as AI hallucination.

Hallucinations
LLMs can “hallucinate,” which means that they tend to confidently generate false responses based on imagined facts. These algorithms can also provide responses that are off-topic if they don’t have an accurate answer to the user’s query, leading to a bad customer experience.

How Does RAG Work?
A typical RAG pipeline has two phases: an offline indexing phase (preparing your data) and a real-time inference phase (answering queries). Here are the key steps.

Step 1: Data collection
You must first gather all the data that is needed for your application. In the case of a customer support chatbot for an electronics company, this can include user manuals, a product database, and a list of FAQs.

Step 2: Data chunking
Data chunking is the process of breaking your data down into smaller, more manageable pieces. For instance, if you have a lengthy 100-page user manual, you might break it down into different sections, each potentially answering different customer questions.

This way, each chunk of data is focused on a specific topic. When a piece of information is retrieved from the source dataset, it is more likely to be directly applicable to the user’s query, since we avoid including irrelevant information from entire documents.

This also improves efficiency, since the system can quickly obtain the most relevant pieces of information instead of processing entire documents.

Step 3: Document embeddings
Now that the source data has been broken down into smaller parts, it needs to be converted into a vector representation. This involves transforming text data into embeddings, which are numeric representations that capture the semantic meaning behind text.

Document embeddings let the system match user queries to relevant information based on meaning rather than exact keyword overlap. A query about “fix my laptop screen” will match a chunk about “display troubleshooting” even though the words differ.

If you’d like to learn more about how text data is converted into vector representations, check out our tutorial on text embeddings with the OpenAI API.

Step 4: Handling user queries
When a user query enters the system, it must also be converted into an embedding or vector representation. The same model must be used for both the document and query embedding to ensure uniformity between the two.

Once the query is converted into an embedding, the system compares the query embedding with the document embeddings. It identifies and retrieves chunks whose embeddings are most similar to the query embedding, using measures such as cosine similarity and Euclidean distance.

These chunks are considered to be the most relevant to the user’s query.

Step 5: Generating responses with an LLM
The retrieved text chunks, along with the initial user query, are fed into a language model. The algorithm will use this information to generate a coherent response to the user’s questions through a chat interface.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/3b7cf5ff-bd8a-4fe5-8a3b-1b5cb7e52ccf" />
























