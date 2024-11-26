import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os

def load_components():
    print("Loading FAISS index...")
    index = faiss.read_index("food_vectors.index")
    print("FAISS index loaded successfully.")

    print("Loading ID to food mapping...")
    with open('id_to_food.json', 'r') as f:
        id_to_food = json.load(f)
    print("ID to food mapping loaded successfully.")

    print("Loading food data...")
    with open('foods_nutrients_map.json', 'r') as f:
        foods_map = json.load(f)
    print("Food data loaded successfully.")

    print("Initializing sentence transformer model...")
    # Force CPU usage and set clear memory management
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = torch.device('cpu')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Sentence transformer model initialized successfully.")

    return index, id_to_food, foods_map, model

def search_foods(model, index, id_to_food, foods_map, query, k=5):
    try:
        # Encode the query with explicit CPU usage
        with torch.no_grad():
            query_vector = model.encode([query], convert_to_numpy=True)
        query_vector = np.array(query_vector).astype('float32')

        # Perform the search
        distances, indices = index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            try:
                food_description = id_to_food[str(int(idx))]
                food_data = foods_map[food_description]
                nutrients = food_data['nutrients']
                serving_info = food_data.get('serving_info', {})

                results.append({
                    'food': food_description,
                    'distance': float(distances[0][i]),
                    'nutrients': nutrients,
                    'serving_info': {
                        'size': serving_info.get('serving_size', 'N/A'),
                        'unit': serving_info.get('serving_unit', 'N/A'),
                        'household': serving_info.get('household_serving', 'N/A')
                    }
                })
            except Exception as e:
                print(f"Error processing result {i}: {e}")
                continue

        return results
    except Exception as e:
        print(f"Error in search_foods: {e}")
        return []

def print_results(results):
    for i, result in enumerate(results):
        try:
            print(f"\n{i+1}. {result['food']}")
            print(f"   Distance: {result['distance']:.4f}")
            print(f"   Serving size: {result['serving_info']['size']} {result['serving_info']['unit']}")
            print(f"   Household serving: {result['serving_info']['household']}")
            print("   Nutrients:")
            for nutrient_name, value in list(result['nutrients'].items())[:5]:
                print(f"      - {nutrient_name}: {value}")
        except Exception as e:
            print(f"Error printing result {i}: {e}")
            continue

def main():
    try:
        # Set memory management
        torch.cuda.empty_cache()

        # Load components
        index, id_to_food, foods_map, model = load_components()
        print("\nAll components loaded successfully. Ready for queries.")

        while True:
            query = input("\nEnter a food query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            results = search_foods(model, index, id_to_food, foods_map, query)
            if results:
                print(f"\nTop 5 results for '{query}':")
                print_results(results)
            else:
                print("No results found.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
