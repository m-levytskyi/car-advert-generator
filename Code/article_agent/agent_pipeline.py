import logging

from typing import List
from article_agent import ArticleAgent
import json


def filter_out_markdown(text: list[str]) -> list[str]:
    """
    Filter out markdown characters from the text.

    Args:
        text (list[str]): The text from which markdown characters need to be filtered out.

    Returns:
        str: The text after filtering out the markdown characters.
    """
    return [t.replace('**', '').replace('`', '').replace('"', '') for t in text]


class AgentPipeline:
    def __init__(self, brand: str, car_type=None, 
                 instruction: str='Write a sensational paragraph for an advertisement about a car brand based on the information provided.'):
        self.brand = brand
        self.car_type = car_type
        self.tasks = self.create_tasks(brand, car_type)
        self.instruction = instruction
        self.agent = ArticleAgent()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      
    
    def create_tasks(self, brand: str, car_type: str=None) -> List[str]:
        """
        Create tasks based on the brand.

        Args:
            brand (str): The brand for which tasks need to be created.

        Returns:
            List[str]: The list of tasks created based on the brand.
        """
        if car_type:
            tasks = [
                f"Write a introductory paragraph about {brand} mentioned in the context of {car_type}. Use a specific car model.",
                f"Descibe a new {car_type} offered by {brand}.",
                f"Explain the history of {brand} mentioned in the context of {car_type}.",
                f"Discuss the innovations of the {car_type} {brand}."
            ]
            return tasks
        
        tasks = tasks = [
            f"Write a introductory paragraph about {brand}.",
            f"Descibe a new car offered by {brand}.",
            f"Explain the history of {brand}.",
            f"Discuss the innovations of the car {brand}."
        ]
        return tasks
    

    # make it return a json object
    def __call__(self):
        logging.info(f"Article Agent: Creating paragraphs, image descriptions and subtitles for the tasks: {self.tasks}")
        paragraphs = filter_out_markdown(self.agent.create_paragraphs(self.tasks, self.instruction, self.brand, self.car_type))
        image_descriptions = filter_out_markdown(self.agent.create_image_descriptions(paragraphs=paragraphs))
        subtitles = filter_out_markdown(self.agent.create_image_subtitles(descriptions=image_descriptions))

        logging.info('Article Agent: Created finished creating paragraphs, image descriptions and subtitles.')
        return json.dumps({
            'paragraphs': paragraphs,
            'captions': subtitles,
            'prompts': image_descriptions,
            'brand': self.brand,
            'car_type': self.car_type
        })


if __name__ == "__main__":
    brand = 'Toyota'
    car_type = 'Coupe'
    agent_pipeline = AgentPipeline(brand=brand, car_type=car_type)
    response = agent_pipeline()
    # save json object to a file

    with open(f"Code/article_agent/json/output_{brand}_{car_type}.json", "w") as f:
        f.write(response)



    