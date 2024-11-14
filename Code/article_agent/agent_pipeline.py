from typing import List
from Code.article_agent.article_agent import ArticleAgent

class AgentPipeline:
    def __init__(self, brand: str, tasks: List[str], instruction: str):
        self.brand = brand
        self.tasks = tasks
        self.instruction = instruction
        self.agent = ArticleAgent(brand)

    # make it return a json object
    def __call__(self):
        paragraphs = self.agent.create_paragraphs(self.tasks, self.instruction)
        image_descriptions = self.agent.create_image_descriptions(paragraphs=paragraphs)
        subtitles = self.agent.create_image_subtitles(descriptions=image_descriptions)
        return {
            'paragraphs': paragraphs,
            'captions': subtitles,
            'propmts': image_descriptions
        }
    