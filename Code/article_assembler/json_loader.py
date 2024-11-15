import json

def load_car_info(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Format the prompts with the car brand and model
    car_brand = data["car_brand"]
    car_model = data["car_model"]
    
    # Replace placeholders in prompts
    formatted_prompts = [
        prompt.format(car_brand=car_brand, car_model=car_model) 
        for prompt in data["content"]["prompts"]
    ]
    
    # Update the prompts in the original data structure
    data["content"]["prompts"] = formatted_prompts
    
    return data
