import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
from .algorithm import create_neural_mark


class ImageAuthor:
    def __init__(self, private_key: str, logger: logging.Logger, config: dict):
        """Initialize an author with their private key."""
        self.private_key = private_key
        self.logger = logger
        self.config = config

        # Set up database path - this can be global or run-specific
        if self.config.get('database', {}).get('run_specific', False):
            self.db_path = Path(self.config['directories']['base_output']) / \
                           self.config.get('run_id', 'default') / 'database'
        else:
            self.db_path = Path(self.config['directories']['database'])

        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize fingerprinter based on selected algorithm
        algorithm_type = config['algorithm_selection']['type']
        self.fingerprinter = create_neural_mark(
            algorithm_type=algorithm_type,
            private_key=private_key,
            logger=logger,
            config=config
        )

        # Initialize fingerprint database
        self.fingerprint_database = {}
        self._load_database()

    def _load_database(self):
        """Load fingerprint database from file if it exists."""
        db_file = self.db_path / 'fingerprint_db.json'
        if db_file.exists():
            try:
                with open(db_file, 'r') as f:
                    db_data = json.load(f)
                    self.fingerprint_database = {
                        k: {
                            'fingerprint': np.array(v['fingerprint']),
                            'original_path': v['original_path'],
                            'timestamp': v['timestamp'],
                            'algorithm': v.get('algorithm', self.config['algorithm_selection']['type'])
                        }
                        for k, v in db_data.items()
                    }
                self.logger.info(f"Loaded {len(self.fingerprint_database)} fingerprints from database")
            except Exception as e:
                self.logger.error(f"Error loading fingerprint database: {str(e)}")
                self.fingerprint_database = {}

    def _save_database(self):
        """Save fingerprint database to file."""
        db_file = self.db_path / 'fingerprint_db.json'
        try:
            # Create directory if it doesn't exist
            db_file.parent.mkdir(parents=True, exist_ok=True)

            db_data = {
                k: {
                    'fingerprint': v['fingerprint'].tolist(),
                    'original_path': v['original_path'],
                    'timestamp': v['timestamp'],
                    'algorithm': v.get('algorithm', self.config['algorithm_selection']['type'])
                }
                for k, v in self.fingerprint_database.items()
            }

            with open(db_file, 'w') as f:
                json.dump(db_data, f)

            self.logger.debug("Fingerprint database saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving fingerprint database: {str(e)}")

    def fingerprint_image(self, image_path: str, output_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Fingerprint an image and store its fingerprint."""
        self.logger.info(f"Fingerprinting image: {image_path}")

        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Generate and embed fingerprint
            fingerprint = self.fingerprinter.generate_fingerprint(img)
            fingerprinted_img = self.fingerprinter.embed_fingerprint(img, fingerprint)

            # Generate comparison fingerprint if enabled
            comparison_results = None
            if self.comparison_mode:
                comparison_fp = self.comparison_fingerprinter.generate_fingerprint(img)
                comparison_img = self.comparison_fingerprinter.embed_fingerprint(img, comparison_fp)
                comparison_results = (comparison_img, comparison_fp)

            # Store fingerprint
            img_hash = self._compute_image_hash(img)
            self.fingerprint_database[img_hash] = {
                'fingerprint': fingerprint,
                'original_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'algorithm': self.config['algorithm_selection']['type']
            }

            # Save fingerprinted image
            cv2.imwrite(output_path, fingerprinted_img)

            # Save comparison image if available
            if comparison_results:
                comparison_path = str(Path(output_path).parent / f"comparison_{Path(output_path).name}")
                cv2.imwrite(comparison_path, comparison_results[0])

            # Update database file
            self._save_database()

            self.logger.info("Image fingerprinted successfully")
            if comparison_results:
                return (fingerprinted_img, fingerprint), comparison_results
            return fingerprinted_img, fingerprint

        except Exception as e:
            self.logger.error(f"Error fingerprinting image: {str(e)}")
            raise

    def verify_ownership(self, suspect_image_path: str) -> Tuple[bool, Optional[str], float, List[str]]:
        """Verify if a suspect image is derived from any of author's original images."""
        self.logger.info(f"Verifying ownership of: {suspect_image_path}")

        try:
            suspect_img = cv2.imread(suspect_image_path)
            if suspect_img is None:
                raise ValueError(f"Could not read suspect image: {suspect_image_path}")

            best_match = None
            highest_similarity = 0
            detected_modifications = []
            used_algorithm = None

            # Extract fingerprint from suspect image
            extracted_fp = self.fingerprinter.extract_fingerprint(suspect_img)

            # Compare with all stored fingerprints
            for img_hash, data in self.fingerprint_database.items():
                # Use appropriate algorithm for verification
                current_algorithm = data.get('algorithm', self.config['algorithm_selection']['type'])
                verifier = self.fingerprinter if current_algorithm == self.config['algorithm_selection'][
                    'type'] else self.comparison_fingerprinter

                is_authentic, similarity, mods = verifier.verify_fingerprint(
                    suspect_img,
                    data['fingerprint']
                )

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = data
                    detected_modifications = mods
                    used_algorithm = current_algorithm

            # Check if it's a match
            threshold = self.config['algorithm']['similarity_threshold']
            is_owned = highest_similarity > threshold

            if is_owned:
                self.logger.info(f"Match found using {used_algorithm}: {best_match['original_path']}")
                return True, best_match['original_path'], highest_similarity, detected_modifications
            else:
                self.logger.info("No match found")
                return False, None, highest_similarity, detected_modifications

        except Exception as e:
            self.logger.error(f"Error verifying ownership: {str(e)}")
            raise

    def _compute_image_hash(self, img: np.ndarray) -> str:
        """Compute perceptual hash of image."""
        try:
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize to 64x64
            img_resized = cv2.resize(img_gray, (64, 64))

            # Compute DCT
            dct = cv2.dct(np.float32(img_resized))

            # Take top-left 8x8 of DCT
            dct_block = dct[:8, :8]

            # Compute median value
            median = np.median(dct_block)

            # Create hash string
            hash_string = ''
            for i in range(8):
                for j in range(8):
                    hash_string += '1' if dct_block[i, j] > median else '0'

            return hash_string

        except Exception as e:
            self.logger.error(f"Error computing image hash: {str(e)}")
            raise
