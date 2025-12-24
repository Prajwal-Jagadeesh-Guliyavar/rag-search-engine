from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    """
    A class for performing multimodal search tasks, starting with image embedding.
    """
    def __init__(self, model_name="clip-ViT-B-32"):
        """
        Initializes the MultimodalSearch with a SentenceTransformer model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
                              Defaults to "clip-ViT-B-32" for CLIP embeddings.
        """
        print(f"Loading SentenceTransformer model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Please ensure the model is available and sentence-transformers is installed.")
            raise

    def embed_image(self, image_path: str):
        """
        Generates an embedding for an image file.

        Args:
            image_path (str): The path to the image file.

        Returns:
            numpy.ndarray: The image embedding.
        """
        print(f"Loading image from: {image_path}")
        try:
            img = Image.open(image_path)
            print("Image loaded.")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            raise
        except Exception as e:
            print(f"Error opening image file {image_path}: {e}")
            raise

        print("Generating image embedding...")
        # SentenceTransformer.encode expects a list of items.
        # For a single image, pass it as a list of one element.
        embeddings = self.model.encode([img])

        # Return the first (and only) embedding
        return embeddings[0]

def verify_image_embedding(image_path: str):
    """
    Generates an embedding for an image and prints its shape.

    Args:
        image_path (str): The path to the image file.
    """
    try:
        # Initialize MultimodalSearch with the default CLIP model
        multimodal_search = MultimodalSearch(model_name="clip-ViT-B-32")

        # Generate the embedding for the image
        embedding = multimodal_search.embed_image(image_path)

        # Print the shape of the embedding
        print(f"Embedding shape: {embedding.shape[0]} dimensions")

    except Exception as e:
        print(f"An error occurred during verification: {e}")

if __name__ == "__main__":
    # This block is for direct execution and testing of the module's functions.
    # For CLI usage, the script in cli/multimodal_search_cli.py will be used.
    print("Running multimodal_search.py directly is not intended for general use.")
    print("Please use the dedicated CLI script: cli/multimodal_search_cli.py")