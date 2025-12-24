import argparse
import os

# Import the verification function from our library module
# Assumes lib.multimodal_search is correctly importable from the project root
from lib.multimodal_search import verify_image_embedding

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for multimodal search operations."
    )

    # Create subparsers to handle different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create the parser for the "verify_image_embedding" command
    verify_parser = subparsers.add_parser(
        'verify_image_embedding',
        help='Verify image embedding generation.'
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file for which to generate an embedding."
    )

    args = parser.parse_args()

    # Execute the appropriate function based on the subcommand
    if args.command == 'verify_image_embedding':
        # Check if the provided image path exists
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found at '{args.image_path}'")
            return

        # Call the verification function from the library module
        try:
            verify_image_embedding(args.image_path)
        except Exception as e:
            print(f"An error occurred: {e}")
    # Add other commands here in the future if needed
    # elif args.command == 'another_command':
    #     ...
    elif args.command is None:
        # If no command is provided, print help
        parser.print_help()

if __name__ == "__main__":
    main()
