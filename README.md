# Langchain-And-Langgraaph--Interview-Questions

LangGraph is a library within the LangChain ecosystem that provides a framework for defining, coordinating, and executing multiple LLM agents (or chains) in a structured and efficient manner.

What Are Vector Embeddings? An Intuitive Explanation
Vector embeddings are numerical representations of words or phrases that capture their meanings and relationships, helping machine learning models understand text more effectively.

What Are Vector Embeddings? 
Vector embeddings are digital fingerprints for words or other pieces of data. Instead of using letters or images, they use numbers that are arranged in a specific structure called a vector, which is like an ordered list of values.
Vector embeddings are a valuable technique for transforming complex data into a format suitable for machine learning algorithms. By converting high-dimensional and categorical data into lower-dimensional, continuous representations, embeddings enhance model performance and computational efficiency while preserving underlying data patterns.


## What Is Retrieval Augmented Generation (RAG)?
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


Step 4: Handling user queries
When a user query enters the system, it must also be converted into an embedding or vector representation. The same model must be used for both the document and query embedding to ensure uniformity between the two.

Once the query is converted into an embedding, the system compares the query embedding with the document embeddings. It identifies and retrieves chunks whose embeddings are most similar to the query embedding, using measures such as cosine similarity and Euclidean distance.

These chunks are considered to be the most relevant to the user’s query.

Step 5: Generating responses with an LLM
The retrieved text chunks, along with the initial user query, are fed into a language model. The algorithm will use this information to generate a coherent response to the user’s questions through a chat interface.

Here is a simplified flowchart summarizing how RAG works:



<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/3b7cf5ff-bd8a-4fe5-8a3b-1b5cb7e52ccf" />


Vector databases like Pinecone, ChromaDB, FAISS, and Weaviate are purpose-built for this problem. They store embeddings and perform fast approximate nearest-neighbor (ANN) search, returning the most relevant chunks in milliseconds even across millions of documents.


## What Is Self-RAG?
Self-RAG introduces iterative reasoning and self-evaluation, allowing the system to dynamically adjust retrieval and generation until a high-quality response is achieved. Instead of treating RAG as a one-shot process, self-RAG incorporates feedback loops to make smarter decisions at every step.

The Self-RAG process involves four key decisions:

Should retrieval happen?
The system first decides whether to retrieve documents based on the question alone or based on an initial answer attempt. This avoids unnecessary retrieval and enables dynamic adaptation.
Are the retrieved documents relevant?
Each retrieved document is independently evaluated for relevance. If a document is not useful for answering the query, it is discarded before generation.
Is the generated response grounded in the retrieved documents?
After generating an answer, the system checks whether all claims in the response are supported by the retrieved content. If unsupported statements are found, the system can attempt to refine the response.
Does the response actually answer the user’s question?
Even if an answer is factually correct, it might not fully address the user’s question. Given the answer and user’s question, the system will predict if the final answer is useful or not (binary) and decide whether to regenerate or stop.
By introducing self-reflection at every step, Self-RAG makes retrieval more reliable, preventing hallucinations and ensuring the final answer is grounded and relevant to the user’s question.


<img width="460" height="545" alt="image" src="https://github.com/user-attachments/assets/cc22b72b-7c24-449b-af5c-dd7310e81eef" />



## What Is Corrective RAG (CRAG)?
Corrective retrieval-augmented generation (CRAG) is an improved version of RAG that aims to make language models more accurate.

While traditional RAG simply uses retrieved documents to help generate text, CRAG takes it a step further by actively checking and refining these documents to ensure they are relevant and accurate. This helps reduce errors or hallucinations where the model might produce incorrect or misleading information.


<img width="1442" height="1186" alt="image" src="https://github.com/user-attachments/assets/a458b35e-16dd-49fc-84a7-2ad014980d3b" />



The CRAG framework operates through a few key steps, which involve a retrieval evaluator and specific corrective actions.

For any given input query, a standard retriever first pulls a set of documents from a knowledge base. These documents are then reviewed by a retrieval evaluator to determine each document's relevance to the query.

In CRAG, the retrieval evaluator is a fine-tuned T5-large model. The evaluator assigns a confidence score to each document, categorizing them into three levels of confidence:

Correct: If at least one document scores above the upper threshold, it is considered correct. The system then applies a knowledge refinement process, using a decompose-then-recompose algorithm to extract the most important and relevant knowledge strips while filtering out any irrelevant or noisy data within the documents. This ensures that only the most accurate and relevant information is retained for the generation process.
Incorrect: If all documents fall below a lower threshold, they are marked as incorrect. In this case, CRAG discards all the retrieved documents and instead performs a web search to gather new, potentially more accurate external knowledge. This step extends the retrieval process beyond static or limited knowledge base by leveraging the vast and dynamic information available on the web, increasing the likelihood of retrieving relevant and accurate data.
Ambiguous: When the retrieved documents contain mixed results, it will be considered ambiguous. In this case, CRAG combines both strategies: it refines information from the initially retrieved documents and incorporates additional knowledge obtained from web searches.
After one of these actions is taken, the refined knowledge is used to generate the final response.

CRAG vs. traditional RAG
CRAG makes several key improvements over traditional RAG. One of its biggest advantages is its ability to fix errors in the information it retrieves. The retrieval evaluator in CRAG helps spot when information is wrong or irrelevant, so it can be corrected before it affects the final output. This means CRAG provides more accurate and reliable information, cutting down on errors and misinformation.

CRAG also excels in making sure the information is both relevant and accurate. While traditional RAG might only check relevance scores, CRAG goes further by refining the documents to ensure they are not just relevant but also precise. It filters out irrelevant details and focuses on the most important points, so the generated text is based on accurate information.


<img width="496" height="1124" alt="image" src="https://github.com/user-attachments/assets/f09420f1-ce28-438b-ba3b-773c77512fa8" />






















