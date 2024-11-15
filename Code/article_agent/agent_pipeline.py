import logging

from typing import List
from article_agent import ArticleAgent


class AgentPipeline:
    def __init__(self, brand: str, car_type=None, 
                 instruction: str='Write a sensational paragraph for an advertisement about a car brand based on the information provided.'):
        self.brand = brand
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
                f"Write a introductory paragraph about {brand} mentioned in the context of {car_type}.",
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
        paragraphs = self.agent.create_paragraphs(self.tasks, self.instruction)
        image_descriptions = self.agent.create_image_descriptions(paragraphs=paragraphs)
        subtitles = self.agent.create_image_subtitles(descriptions=image_descriptions)
        logging.info('Article Agent: Created finished creating paragraphs, image descriptions and subtitles.')
        print(f"Article Agent: \nParagraphs: {paragraphs}, \n\nImage Descriptions: {image_descriptions}, \n\nImage Subtitles: {subtitles}")
        return {
            'paragraphs': paragraphs,
            'captions': subtitles,
            'propmts': image_descriptions
        }

if __name__ == "__main__":
    agent_pipeline = AgentPipeline(brand='BMW', car_type='SUV')
    response = agent_pipeline()


    