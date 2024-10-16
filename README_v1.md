# AlphaPunch: Robust Image Fingerprinting Algorithm

AlphaPunch is a novel image fingerprinting technique that utilizes the alpha channel and DCT (Discrete Cosine Transform) to embed a unique and verifiable fingerprint in images. This algorithm is designed to be robust against common image transformations while maintaining image quality.

## Features

- Secure fingerprint generation using a private key
- DCT-based embedding in the YCbCr color space
- Adaptive embedding strength based on image content
- Error correction using repetition code
- Robust against common image transformations (compression, resizing, cropping)
- Image quality assessment using PSNR and SSIM metrics

## Requirements

- Python 3.7+
- numpy
- Pillow
- scipy
- pycryptodome
- tqdm
- opencv-python
- scikit-image

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/alphapunch.git
   cd alphapunch
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Embedding a Fingerprint

```python
from algorithm import AlphaPunch

alphapunch = AlphaPunch(private_key="your_secret_key_here")
salt = alphapunch.embed_fingerprint("input_image.jpg", "fingerprinted_image.png")
```

### Verifying a Fingerprint

```python
is_authentic, similarity = alphapunch.verify_fingerprint("fingerprinted_image.png", salt)
print(f"Image is {'authentic' if is_authentic else 'not authentic'}")
print(f"Similarity: {similarity:.2%}")
```

### Running Tests with Unsplash Images

1. Get an Unsplash API key from https://unsplash.com/developers
2. Replace `YOUR_UNSPLASH_ACCESS_KEY` in `unsplash_tester.py` with your API key
3. Run the tester:
   ```
   python unsplash_tester.py --num_images 20
   ```

## Algorithm Details

AlphaPunch uses the following techniques:
1. YCbCr color space conversion for embedding in the luminance channel
2. DCT-based coefficient modification for robust embedding
3. Adaptive embedding strength based on local image characteristics
4. Simple repetition code for error correction
5. Secure fingerprint generation using a private key and salt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by various digital watermarking techniques
- Uses the Unsplash API for testing with diverse image sets