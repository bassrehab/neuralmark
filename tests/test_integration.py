import unittest

from tests.test_base import AlphaPunchTestBase
from alphapunch.author import ImageAuthor
import numpy as np
import cv2
import os
import time
from pathlib import Path


class TestIntegration(AlphaPunchTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.author = ImageAuthor(
            private_key=cls.config['private_key'],
            logger=cls.logger,
            config=cls.config
        )

    def test_full_pipeline(self):
        """Test complete fingerprinting pipeline."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                # Save test image
                test_path = os.path.join('output', f'test_{name}.png')
                cv2.imwrite(test_path, image)

                # Fingerprint image
                output_path = os.path.join('output', f'fp_{name}.png')
                fingerprinted_img, fingerprint = self.author.fingerprint_image(
                    test_path, output_path
                )

                # Verify original
                is_owned, orig_path, similarity, mods = self.author.verify_ownership(output_path)
                self.assertTrue(is_owned)
                self.assertGreater(similarity, self.config['algorithm']['similarity_threshold'])

                # Test with modifications
                modified_path = os.path.join('output', f'mod_{name}.png')
                modified = cv2.GaussianBlur(fingerprinted_img, (3, 3), 0)
                cv2.imwrite(modified_path, modified)

                is_owned, orig_path, similarity, mods = self.author.verify_ownership(modified_path)
                self.assertTrue(is_owned)  # Should still recognize modified version

    def test_batch_processing(self):
        """Test system with batch of images."""
        # Create batch of test images
        batch_size = 5
        test_batch = []
        for i in range(batch_size):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            path = os.path.join('output', f'batch_{i}.png')
            cv2.imwrite(path, img)
            test_batch.append(path)

        # Process batch
        results = []
        for img_path in test_batch:
            output_path = os.path.join('output', f'fp_{os.path.basename(img_path)}')
            fingerprinted_img, fingerprint = self.author.fingerprint_image(img_path, output_path)
            results.append((output_path, fingerprint))

        # Verify all results
        for output_path, _ in results:
            is_owned, _, similarity, _ = self.author.verify_ownership(output_path)
            self.assertTrue(is_owned)

    def test_cross_device_verification(self):
        """Test verification across different simulated devices/conditions."""
        # Create base test image
        test_image = self.test_images['pattern']
        base_path = os.path.join('output', 'base.png')
        cv2.imwrite(base_path, test_image)

        # Fingerprint base image
        fp_path = os.path.join('output', 'fp_base.png')
        _, fingerprint = self.author.fingerprint_image(base_path, fp_path)

        # Test different conditions
        conditions = {
            'brightness': lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=10),
            'contrast': lambda x: cv2.convertScaleAbs(x, alpha=1.3, beta=0),
            'quality': lambda x: cv2.imdecode(
                cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, 85])[1],
                cv2.IMREAD_COLOR
            ),
            'resize': lambda x: cv2.resize(cv2.resize(x, (512, 512)), (256, 256)),
            'screenshot': lambda x: cv2.GaussianBlur(x, (3, 3), 0),  # Simulate screenshot
            'crop': lambda x: x[10:-10, 10:-10]
        }

        base_img = cv2.imread(fp_path)
        for cond_name, condition_func in conditions.items():
            # Apply condition
            modified = condition_func(base_img.copy())
            mod_path = os.path.join('output', f'condition_{cond_name}.png')
            cv2.imwrite(mod_path, modified)

            # Verify
            is_owned, _, similarity, mods = self.author.verify_ownership(mod_path)
            self.assertTrue(is_owned, f"Failed verification under condition: {cond_name}")

    def test_database_persistence(self):
        """Test fingerprint database persistence and recovery."""
        # Generate and store some fingerprints
        test_fingerprints = {}
        for name, image in self.test_images.items():
            test_path = os.path.join('output', f'db_test_{name}.png')
            cv2.imwrite(test_path, image)

            fp_path = os.path.join('output', f'db_fp_{name}.png')
            _, fingerprint = self.author.fingerprint_image(test_path, fp_path)
            test_fingerprints[name] = fingerprint

        # Create new author instance (simulating restart)
        new_author = ImageAuthor(
            private_key=self.config['private_key'],
            logger=self.logger,
            config=self.config
        )

        # Verify fingerprints are recovered
        for name in test_fingerprints:
            test_path = os.path.join('output', f'db_fp_{name}.png')
            is_owned, _, similarity, _ = new_author.verify_ownership(test_path)
            self.assertTrue(is_owned, f"Failed to recover fingerprint for {name}")

    def test_concurrent_operations(self):
        """Test system behavior with concurrent operations."""
        import threading

        def process_image(img_name):
            img = self.test_images[img_name]
            test_path = os.path.join('output', f'concurrent_{img_name}.png')
            cv2.imwrite(test_path, img)

            fp_path = os.path.join('output', f'concurrent_fp_{img_name}.png')
            self.author.fingerprint_image(test_path, fp_path)

            is_owned, _, _, _ = self.author.verify_ownership(fp_path)
            self.assertTrue(is_owned)

        # Run concurrent operations
        threads = []
        for img_name in self.test_images:
            thread = threading.Thread(target=process_image, args=(img_name,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

    def test_error_handling(self):
        """Test system error handling and recovery."""
        # Test with invalid image
        with self.assertRaises(Exception):
            self.author.fingerprint_image(
                'nonexistent.png',
                'output.png'
            )

        # Test with corrupted image
        corrupt_path = os.path.join('output', 'corrupt.png')
        with open(corrupt_path, 'wb') as f:
            f.write(b'corrupted data')

        with self.assertRaises(Exception):
            self.author.verify_ownership(corrupt_path)

        # Test recovery after errors
        test_path = os.path.join('output', 'recovery_test.png')
        cv2.imwrite(test_path, self.test_images['pattern'])

        # System should still work after handling errors
        fp_path = os.path.join('output', 'recovery_fp.png')
        _, fingerprint = self.author.fingerprint_image(test_path, fp_path)
        is_owned, _, _, _ = self.author.verify_ownership(fp_path)
        self.assertTrue(is_owned)

    def test_performance_metrics(self):
        """Test system performance metrics."""
        metrics = {}
        test_image = self.test_images['pattern']
        test_path = os.path.join('output', 'perf_test.png')
        cv2.imwrite(test_path, test_image)

        # Measure fingerprinting time
        start_time = time.time()
        fp_path = os.path.join('output', 'perf_fp.png')
        _, _ = self.author.fingerprint_image(test_path, fp_path)
        metrics['fingerprint_time'] = time.time() - start_time

        # Measure verification time
        start_time = time.time()
        _, _, _, _ = self.author.verify_ownership(fp_path)
        metrics['verify_time'] = time.time() - start_time

        # Check performance thresholds
        self.assertLess(metrics['fingerprint_time'], 2.0)  # Should take less than 2 seconds
        self.assertLess(metrics['verify_time'], 1.0)  # Should take less than 1 second


if __name__ == '__main__':
    unittest.main()