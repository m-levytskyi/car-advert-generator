import os
from dotenv import load_dotenv
from typing import Optional, List
import wikipediaapi
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class ArticleAgent:
    def __init__(self, repo_id: str = "HuggingFaceH4/zephyr-7b-beta", task: str = "text-generation", max_new_tokens: int = 512, 
                 do_sample: bool = False, repetition_penalty: float = 1.03, 
                 user_agent: str = "MyArticleAgent/1.0 (http://mywebsite.com; contact@myemail.com)"):
        # Load environment variables from .env file
        load_dotenv()

        # Get the Hugging Face token from the environment variables
        self.hf_token: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

        # Initialize the language model
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task=task,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        
        # Initialize the chat model
        self.chat_model = ChatHuggingFace(llm=self.llm)

        # Initialize the Wikipedia retriever
        self.retriever = WikipediaRetriever()

        # Set up the prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """
            Perform the task based only on the context provided.
            Context: {context}
            Task: {task}
            """
        )

    def get_information(self, task: str) -> str:
        """Retrieves context from Wikipedia and generates an answer to the given task."""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Define the processing chain
        # Retrieval: The self.retriever fetches documents based on the task (query). The | operator is used here as a way to combine different processing steps, which is a common pattern in LangChain.
        # Formatting: The retrieved documents are then formatted into a string representation using format_docs.
        # Prompt Preparation: The formatted context is inserted into the prompt template using the self.prompt.
        # Generation: Finally, the formatted prompt is passed to the language model (self.llm), which generates the final answer.
        # Output Parsing: The output is then parsed to extract the response using StrOutputParser.
        chain = (
            {"context": self.retriever | format_docs, "task": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Generate the answer
        return chain.invoke(task)

agent = ArticleAgent()
answer = agent.get_information('Tell me something about the brand BMW.')
print(answer)
