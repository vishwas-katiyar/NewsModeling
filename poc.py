from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_title_and_topic(article_text):
    # Format the prompt with instructions
    prompt = f"Generate a concise title and a brief topic description for the following news article:\n\n{article_text}\n\nTitle and Topic:"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    
    # Generate output
    outputs = model.generate(
        inputs['input_ids'],
        max_length=512,
        num_return_sequences=1,
        early_stopping=True,
        temperature=0.7,  # Adjust for more or less randomness
        top_p=0.9  # Adjust for more or less randomness
    )
    
    # Decode the output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract title and topic
    # Assuming output is formatted with "Title:" and "Topic:" prefixes
    title_start = result.find("Title:") + len("Title:")
    title_end = result.find("Topic:")
    topic_start = title_end + len("Topic:")
    
    title = result[title_start:title_end].strip()
    topic = result[topic_start:].strip()
    
    return title, topic

# Example news article
article_text = """
The latest advancements in artificial intelligence are revolutionizing various industries. From healthcare to finance, AI technologies are being integrated to improve efficiency and drive innovation. The development of advanced neural networks and machine learning algorithms is at the forefront of this transformation.
"""

# Generate title and topic
title, topic = generate_title_and_topic(article_text)
print(f"Title: {title}")
print(f"Topic: {topic}")
