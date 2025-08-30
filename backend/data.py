import requests

def download_imagenet_classes(url: str, output_file: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for HTTP errors
    except requests.RequestException as e:
        print(f"Error downloading from {url}: {e}")
        return

    content = response.text

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully saved content to '{output_file}'.")
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}")

if __name__ == "__main__":
    url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    output_file = "imagenet_classes.txt"
    download_imagenet_classes(url, output_file)

