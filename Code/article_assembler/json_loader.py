import json

def load_car_info(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Format the prompts with the car brand and model
    car_brand = 'BMW' #data["car_brand"]
    car_model = 'X5' #data["car_model"]
    
    # Replace placeholders in prompts
    formatted_prompts = [
        prompt.format(car_brand=car_brand, car_model=car_model) 
        for prompt in data["prompts"]
    ]
    
    # Update the prompts in the original data structure
    data["prompts"] = formatted_prompts
    
    return data
