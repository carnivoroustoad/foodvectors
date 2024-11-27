import requests
import json
import numpy as np
from tqdm import tqdm
import time
from typing import List, Dict, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import signal
import pickle
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLOUDFLARE_ACCOUNT_ID = "1ecbd7f1fe2fdd99f2ad145c785dbbdf"
CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')
if not CLOUDFLARE_API_TOKEN:
    raise ValueError("CLOUDFLARE_API_TOKEN environment variable is not set")
WORKER_URL = "https://black-bonus-cf17.carnivoroustoad.workers.dev"

BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 2
EMBEDDING_BATCH_SIZE = 20

class GracefulExit(Exception):
    pass

class CloudflareUploader:
    def __init__(self):
        self.headers = {
            'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}',
            'Content-Type': 'application/json'
        }
        self.stop_requested = False
        self.progress_file = 'upload_progress.pkl'
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        logger.info("\nGraceful shutdown requested. Finishing current batch...")
        self.stop_requested = True

    def save_progress(self, current_index: int, failed_batches: List[int]):
        progress = {
            'current_index': current_index,
            'failed_batches': failed_batches,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.progress_file, 'wb') as f:
            pickle.dump(progress, f)
        logger.info(f"Progress saved at index {current_index}")

    def load_progress(self) -> Dict:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'rb') as f:
                progress = pickle.load(f)
            logger.info(f"Resuming from index {progress['current_index']}")
            return progress
        return {'current_index': 0, 'failed_batches': [], 'timestamp': None}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ValueError)
    )
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        try:
            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/baai/bge-base-en-v1.5",
                headers=self.headers,
                json={"text": texts},
                timeout=30
            )

            if response.status_code == 403:
                raise ValueError("AI API access forbidden - check API token permissions")

            response.raise_for_status()
            result = response.json()

            if not result.get('success', False):
                errors = result.get('errors', [])
                messages = result.get('messages', [])
                raise ValueError(f"AI API error: {errors or messages}")

            return np.array(result['result']['data'])
        except Exception as e:
            raise ValueError(f"API request failed: {str(e)}")
    def load_data(self) -> Dict[str, Any]:
        """Load food data from the JSON file"""
        try:
            with open('foods_nutrients_map.json', 'r') as f:
                foods_map = json.load(f)
            logger.info(f"Loaded {len(foods_map)} food items")
            return foods_map
        except FileNotFoundError:
            logger.error("foods_nutrients_map.json not found. Run analysis.py first to generate the file.")
            return {}
        except json.JSONDecodeError:
            logger.error("Error decoding foods_nutrients_map.json. The file may be corrupted.")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            return {}
    def process_batch(self,
                     texts: List[str],
                     metadata: List[Dict],
                     start_idx: int) -> bool:
        if self.stop_requested:
            raise GracefulExit("Stop requested")

        try:
            embeddings = []
            failed_indices = []

            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                if self.stop_requested:
                    raise GracefulExit("Stop requested")

                chunk = texts[i:i + EMBEDDING_BATCH_SIZE]
                chunk_indices = list(range(i, min(i + EMBEDDING_BATCH_SIZE, len(texts))))

                try:
                    chunk_embeddings = self.get_embeddings_batch(chunk)
                    embeddings.extend(chunk_embeddings)
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for chunk: {e}")
                    failed_indices.extend(chunk_indices)
                    zero_embeddings = np.zeros((len(chunk), 384))
                    embeddings.extend(zero_embeddings)

                time.sleep(0.1)

            payload_vectors = []
            for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                if i not in failed_indices:
                    payload_vectors.append({
                        'id': str(start_idx + i),
                        'values': embedding.tolist(),
                        'metadata': meta
                    })

            if not payload_vectors:
                logger.warning(f"No valid vectors in batch starting at index {start_idx}")
                return True

            for attempt in range(MAX_RETRIES):
                if self.stop_requested:
                    raise GracefulExit("Stop requested")

                try:
                    response = requests.post(
                        f"{WORKER_URL}/upload",
                        json={'vectors': payload_vectors},
                        headers=self.headers,
                        timeout=30
                    )
                    response.raise_for_status()
                    return True
                except Exception as e:
                    logger.error(f"Upload attempt {attempt + 1} failed: {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        logger.error(f"Failed to upload batch starting at index {start_idx}")
                        return False

            return False

        except GracefulExit:
            raise
        except Exception as e:
            logger.error(f"Error processing batch starting at {start_idx}: {e}")
            return False

    def upload_to_cloudflare(self):
        try:
            foods_map = self.load_data()
            if not foods_map:
                logger.error("No data to upload")
                return False

            # Load progress
            progress = self.load_progress()
            start_index = progress['current_index']
            failed_batches = progress['failed_batches']

            if start_index > 0:
                logger.info(f"Resuming upload from index {start_index}")
                if failed_batches:
                    logger.info(f"Previously failed batches: {failed_batches}")

            food_descriptions = list(foods_map.keys())[start_index:]
            total_items = len(food_descriptions)

            logger.info(f"Total remaining items to process: {total_items}")

            successful_batches = 0
            total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE

            with tqdm(total=total_batches, desc="Processing batches") as pbar:
                try:
                    for batch_idx in range(total_batches):
                        current_start_idx = start_index + (batch_idx * BATCH_SIZE)
                        end_idx = min(current_start_idx + BATCH_SIZE, start_index + total_items)

                        batch_descriptions = list(foods_map.keys())[current_start_idx:end_idx]
                        batch_metadata = [
                            {
                                'description': desc,
                                'nutrients': foods_map[desc]['nutrients'],
                                'serving_info': foods_map[desc].get('serving_info', {}),
                                'brand_info': foods_map[desc].get('brand_info', {})
                            }
                            for desc in batch_descriptions
                        ]

                        if self.process_batch(batch_descriptions, batch_metadata, current_start_idx):
                            successful_batches += 1
                        else:
                            failed_batches.append(current_start_idx)

                        # Save progress after each batch
                        self.save_progress(end_idx, failed_batches)

                        pbar.update(1)
                        time.sleep(0.5)

                except GracefulExit:
                    logger.info("\nGracefully stopping... Progress has been saved.")
                    return

            # Final statistics
            success_rate = (successful_batches / total_batches) * 100
            logger.info(f"Upload complete. Success rate: {success_rate:.2f}%")
            logger.info(f"Successfully processed {successful_batches} out of {total_batches} batches")

            if failed_batches:
                logger.warning(f"Failed batches starting at indices: {failed_batches}")
                logger.warning("You can resume uploading to retry these batches")

            # Clean up progress file if complete
            if os.path.exists(self.progress_file) and not failed_batches:
                os.remove(self.progress_file)

        except Exception as e:
            logger.error(f"An error occurred during upload: {str(e)}")
            raise

def main():
    uploader = CloudflareUploader()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup-only', action='store_true',
                       help='Only cleanup previous uploads without uploading new data')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Only retry previously failed batches')
    args = parser.parse_args()

    if args.cleanup_only:
        uploader.cleanup_previous_uploads()
    else:
        uploader.upload_to_cloudflare()
        if not uploader.stop_requested:
            uploader.validate_upload()

if __name__ == "__main__":
    main()
