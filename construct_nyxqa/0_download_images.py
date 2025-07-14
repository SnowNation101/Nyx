import os
import requests
import json
from datasets import load_dataset
from tqdm import tqdm


data_dir = "/fs/archive/share/mm_datasets/OBELICS"
obelics_dataset = load_dataset(data_dir)


def download_image_from_url(image_url: str, save_path: str, timeout: int = 10) -> bool:
    """
    Downloads an image from the given URL.
    Prints an error message if the image does not exist (404 error)
    or if the download does not complete within the specified timeout.
    Returns True if the download is successful, False otherwise.

    Args:
        image_url (str): The URL of the image to download.
        save_path (str): The local path where the image will be saved,
                         including the filename and extension.
        timeout (int): The download timeout in seconds. If the download
                       exceeds this time, an error will be raised.
    """
    print(f"Trying to download image from URL: {image_url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    }

    try:
        response = requests.get(image_url, stream=True, timeout=timeout, headers=headers) 
        
        # Raise an HTTPError for bad responses (4xx or 5xx status codes).
        response.raise_for_status() 

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image successfully downloaded to: {save_path}")
        return True  # Return True if the image is successfully downloaded

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors, specifically checking for 404 Not Found.
        if e.response.status_code == 403: # Specifically check for 403
            print(f"Error: Access forbidden (403 Forbidden) - {image_url}. This might be due to website security measures. Try using appropriate User-Agent and Referer headers.")
        elif e.response.status_code == 404:
            print(f"Error: Image not found or invalid URL (404 Not Found) - {image_url}")
        else:
            print(f"An HTTP error occurred during download (Status Code: {e.response.status_code}): {e}")
    except requests.exceptions.ConnectionError as e:
        # Handle network connection errors.
        print(f"Error: Could not connect to the server. Please check your network connection or URL - {e}")
    except requests.exceptions.Timeout as e:
        # Handle download timeout errors.
        print(f"Error: Download timed out. Exceeded {timeout} seconds - {e}")
    except requests.exceptions.RequestException as e:
        # Handle any other requests-related errors.
        print(f"An unknown error occurred during download: {e}")
    except Exception as e:
        # Handle any other unexpected errors.
        print(f"An unexpected error occurred: {e}")
    
    return False  # Return False if the download fails


total_len = len(obelics_dataset["train"])
save_dir = "/fs/archive/share/mm_datasets/obelics_images"
output_path = "/fs/archive/share/mm_datasets/obelics_processed.jsonl"
image_idx = 0
step_size = 5000

with open(output_path, "w") as f_out: 
    for i in tqdm(range(0, total_len, step_size), desc="Processing dataset entries", unit="entry"):
        print(f"== Processing entry {i / step_size}/{total_len / step_size}...== ")
        data = obelics_dataset["train"][i]
        new_images = []
        new_text = ""
        skip_entry = False
        downloaded_images = []
        first_downloaded_image_idx = None

        for image_url in data["images"]:
            if image_url is not None:
                save_path = os.path.join(save_dir, str(image_idx) + ".jpg")
                flag = download_image_from_url(image_url=image_url, 
                                               save_path=save_path, 
                                               timeout=10)

                if not flag:
                    print(f"Skipping entry {i} due to failed image download: {image_url}")
                    for img in downloaded_images:
                        if os.path.exists(img):
                            os.remove(img)
                    if first_downloaded_image_idx is not None:
                        image_idx = first_downloaded_image_idx
                    skip_entry = True
                    break
                else:
                    new_images.append(str(image_idx) + ".jpg")
                    downloaded_images.append(save_path)
                    if first_downloaded_image_idx is None:
                        first_downloaded_image_idx = image_idx
                    image_idx += 1

        if not skip_entry:
            for text in data["texts"]:
                if text is None:
                    new_text += "<|image|>"
                else:
                    new_text += text

            new_entry = {
                "text": new_text,
                "images": new_images,
                "source_data": data
            }
            
            json.dump(new_entry, f_out)
            f_out.write('\n')
            print("This entry has been processed and saved.")

print(f"Processed data has been saved line by line to {output_path}")