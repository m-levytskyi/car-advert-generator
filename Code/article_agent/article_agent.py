import os
import logging
from dotenv import load_dotenv
from typing import Optional, List
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.tools.render import render_text_description


from agent_tools import search_duckduckgo, search_wikipedia
from langchain.agents import AgentExecutor


class ArticleAgent:
    def __init__(self, list_of_tools: list = [search_wikipedia, search_duckduckgo]):
        """
        Initialize the ArticleAgent with the large language model, Wikipedia retriever, and DuckDuckGo search.
        """
        # Load environment variables
        load_dotenv()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Get the API key for the Groq language model
        self.gq_token: Optional[str] = os.getenv("GROQ_API_KEY")
        self.langsmith_token: Optional[str] = os.getenv("LANGSMITH_API_KEY")

        # Initialize the Groq language model
        self.gr_llm = ChatGroq(
            api_key=self.gq_token,
            model="llama3-8b-8192",
            # model="llama-3.3-70b-specdec", # better model but slower and has a lower token limit
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.hf_token: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

        # Initialize the language model
        self.hf_llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03,
        )

        self.tools = list_of_tools

        # setup tools
        tools = list_of_tools

        # setup ReAct style prompt
        prompt = hub.pull("hwchase17/react-json")
        prompt = prompt.partial(
            tools=render_text_description(tools),
            tool_names=", ".join([t.name for t in tools]),
        )


        # define the agent
        chat_model_with_stop = self.hf_llm.bind(stop=["\nObservation"])
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            }
            | prompt
            | chat_model_with_stop
            | ReActJsonSingleInputOutputParser()
        )

        # instantiate AgentExecutor
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, max_iterations=4, return_intermediate_steps=True)

    def token_in_string(self, string: str) -> int:
        """
        Count the number of tokens in a string.

        Args:
            string (str): The string for which the number of tokens need to be counted.

        Returns:
            int: The number of tokens in the string.
        """
        # count number of words in the string
        return len(string.split()) * 0.75 # 1 token = 0.75 words (average)

    def get_information_with_agent(self, task: str = "", prior_responses: str = "") -> AIMessage:
        try:
            response = self.agent_executor.invoke({"input": task})
            intermediate_steps = response['intermediate_steps']
            all_context = ""
            for step in intermediate_steps:
                context = step[1]
                # Respect the token limit of the model
                if self.token_in_string(all_context + context) > 4096:
                    break
                all_context += context + "\n"

            # call basic self.llm, with all_context as context, systemmessage, and humanmessage (task)
            prompt = ChatPromptTemplate([
                SystemMessage("You are an article writer. You have to write an article given a specific task. Always answer in this format:\"Paragraph: ...\""),
                HumanMessage(f"For context: {all_context}"),
                HumanMessage(f"Prior Paragraphs: {prior_responses}"),
                HumanMessage(task),
            ]).format_prompt()
            return self.gr_llm.invoke(prompt.to_messages())
        except Exception as e:
            logging.error(f"Error in paragraph generation: {e}")
            return None


    def get_information_with_sources_fallback(self, brand: None, type: None, task: str, instruction: str = "Perform the task based only on the context provided. Look at the prior responses as a reference.", prior_responses: str = "") -> AIMessage:
        """
        Get information based on the task and instruction provided, using all tools.

        Args:
            task (str): The task to be performed.
            instruction (str): The instruction to be displayed to the model.
            prior_responses (str): The prior responses to be used as reference.

        Returns:
            str: The answer generated by the model.
        """
        try:
            tool_res_0 = self.tools[0].invoke({"searchstring": brand + " " + type})
            tool_res_1 = self.tools[1].invoke({"searchstring": brand + " " + type})
            

            prompt = f"{instruction} \nPrior Responses: {prior_responses} \nContext: \n{tool_res_0}\n{tool_res_1} \nTask: {task}"
            

            return self.gr_llm.invoke(prompt)
        except Exception as e:
            logging.error(f"Error in paragraph generation fallback: {e}")
            return None
    
    def create_paragraphs(self, tasks: List[str], brand: str = None, car_type: str = None) -> List[str]:
        """
        Create paragraphs for the given tasks.

        Args:
            tasks (List[str]): The list of tasks for which paragraphs need to be created.

        Returns:
            List[str]: The list of paragraphs created for the given tasks.
        """
        paragraphs = []
        responses = ""
        for task in tasks:
            paragraph = self.get_information_with_agent(task=task, prior_responses=responses)
            if paragraph is None:
                logging.info("Model failed to generate a response. Fallback to using all tools.")
                paragraph = self.get_information_with_sources_fallback(brand, car_type, task, prior_responses=responses)
            if paragraph is not None:
                paragraphs.append(paragraph.content)
                responses += paragraph.content + "\n"
        return paragraphs
    
    def get_information(self, context: str, instruction: str) -> str:
        """
        Get information based on the task and instruction provided.

        Args:
            context (str): The context to be used for generating the answer.
            instruction (str): The instruction to be displayed to the model.

        Returns:
            str: The answer generated by the model.
        """
        
        prompt = ChatPromptTemplate.from_template(
            f"""
            {instruction}
            Context: {{context}}
            """
        )

        # Define the processing chain
        chain = (
            {"context": RunnablePassthrough()}
            | prompt
            | self.gr_llm
            | StrOutputParser()
        )

        # Generate the answer
        return chain.invoke(context)

    def create_image_descriptions(self, paragraphs: List[str]) -> List[str]:
        """
        Create descriptions for the given images based on a string.

        Args:
            paragraphs (List[str]): The list of paragraphs based on which the image descriptions need to be created.

        Returns:
            List[str]: The list of image descriptions created for the given paragraphs.
        """
        image_descriptions = []
        views = ["front", "back", "side", "interior"]
        for i, paragraph in enumerate(paragraphs):
            view = views[i % len(views)]
            # for a diffusion model, create the description of the image based on the paragraph
            image_description = self.get_information(context=paragraph, instruction=f"Create a description of an image based on the context provided. Only describe the image. The image should be a {str(view)} view of the car.")
            image_descriptions.append(image_description)
        return image_descriptions

    def create_image_subtitles(self, descriptions: List[str]) -> List[str]:
        """
        Create image subtitles based on image descriptions

        Args:
            descriptions (List[str]): The list of paragraphs based on which the image descriptions need to be created.

        Returns:
            List[str]: The list of image descriptions created for the given paragraphs.
        """
        image_descriptions = []
        for description in descriptions:
            # for a diffusion model, create the description of the image based on the paragraph
            image_description = self.get_information(context=description, instruction="Create a subtitle for an image based on the image description provided in the context. Only give me the subtitle.")
            image_descriptions.append(image_description)
        return image_descriptions

if __name__ == "__main__":
    tools = [search_duckduckgo, search_wikipedia]
    agent = ArticleAgent(list_of_tools=tools)
    brand = 'BMW'
    car_type = 'Coupe'
    tasks = [
        f"Write a introductory paragraph (about 200 words) for an article about a {brand} {car_type}.",
        f"Write a paragraph for an article about a new {car_type} offered by {brand}.",
        f"Write a paragraph for an article about the history of {brand}.",
        f"Write a paragraph for an article about the innovations of the {car_type} of {brand}."
    ]
    using_agent = agent.get_information_with_agent(tasks[0])
    print(using_agent)
    # paragraphs = agent.create_paragraphs(tasks, brand, car_type)
    # for i, paragraph in enumerate(paragraphs):
    #     print(f"\n\nTask: {tasks[i]}\nParagraph: {paragraph}")