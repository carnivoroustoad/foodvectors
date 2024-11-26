import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the food data
with open('foods_nutrients_map.json', 'r') as f:
    foods_map = json.load(f)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare data for vectorization
food_descriptions = list(foods_map.keys())
food_ids = list(range(len(food_descriptions)))

print(f"Generating embeddings for {len(food_descriptions)} food items...")

# Generate embeddings
embeddings = model.encode(food_descriptions, show_progress_bar=True)

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

print(f"Adding vectors to the index...")

# Add vectors to the index
index.add(embeddings.astype('float32'))

# Save the index to disk
faiss.write_index(index, "food_vectors.index")

# Save mapping of IDs to food descriptions
id_to_food = dict(zip(food_ids, food_descriptions))
with open('id_to_food.json', 'w') as f:
    json.dump(id_to_food, f)

print(f"Vector database created with {len(food_descriptions)} entries")
print(f"Vector dimension: {dimension}")

# Example search function
def search_foods(query, k=5):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        food_description = id_to_food[int(idx)]
        food_data = foods_map[food_description]
        nutrients = food_data['nutrients']
        serving_info = food_data.get('serving_info', {})  # Get serving info if available

        result = {
            'food': food_description,
            'distance': float(distances[0][i]),
            'nutrients': nutrients,
            'serving_info': {
                'size': serving_info.get('serving_size', 'N/A'),
                'unit': serving_info.get('serving_unit', 'N/A'),
                'household': serving_info.get('household_serving', 'N/A')
            }
        }
        results.append(result)

    return results

# Modified example usage
print("\nExample search:")
query = "butter chicken"
results = search_foods(query)
print(f"Top 5 results for '{query}':")
for i, result in enumerate(results):
    print(f"{i+1}. {result['food']}")
    print(f"   Distance: {result['distance']:.4f}")
    print(f"   Serving size: {result['serving_info']['size']} {result['serving_info']['unit']}")
    print(f"   Household serving: {result['serving_info']['household']}")
    print(f"   Sample nutrients: {list(result['nutrients'].items())[:3]}")
    print()

# Example usage
print("\nExample search:")
query = "apple"
results = search_foods(query)
print(f"Top 5 results for '{query}':")
for i, result in enumerate(results):
    print(f"{i+1}. {result['food']} (Distance: {result['distance']:.4f})")
    print(f"   Sample nutrients: {list(result['nutrients'].items())[:3]}")
    print()
