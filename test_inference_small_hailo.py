import argparse
from PIL import Image
from inference_small_hailo import run_small_inference


def main():
    parser = argparse.ArgumentParser(description="Test Hailo VLM inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image (jpg/png)")
    parser.add_argument("--trigger", type=str, required=True, help="Trigger description (e.g., 'a dog on a couch')")
    parser.add_argument("--model-dir", type=str, default="./hefs_and_embeddings", help="Model directory (default: ./hefs_and_embeddings)")
    args = parser.parse_args()

    # Load image
    try:
        image = Image.open(args.image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Run inference
    print(f"Running inference on: {args.image}")
    print(f"Trigger: {args.trigger}")
    result = run_small_inference(
        frame=image,
        trigger_description=args.trigger,
        model_dir=args.model_dir
    )
    print(f"Result: {result}")

    # second Run inference
    args.trigger = "a dog on a couch"
    print(f"Second Running inference on: {args.image}")
    print(f"Trigger: {args.trigger}")
    result = run_small_inference(
        frame=image,
        trigger_description=args.trigger,
        model_dir=args.model_dir
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    main()