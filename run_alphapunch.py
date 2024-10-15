import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
from alphapunch.algorithm import EnhancedAlphaPunch


def setup_logger():
    logger = logging.getLogger('AlphaPunch')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('alphapunch.log')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="AlphaPunch Image Fingerprinting")
    parser.add_argument('--input', type=str, help="Input image path")
    parser.add_argument('--output', type=str, help="Output image path")
    parser.add_argument('--embed', action='store_true', help="Embed fingerprint")
    parser.add_argument('--verify', action='store_true', help="Verify fingerprint")
    parser.add_argument('--gui', action='store_true', help="Launch GUI")
    parser.add_argument('--embed_strength', type=float, default=0.75, help="Embedding strength")
    parser.add_argument('--fingerprint_size', type=int, nargs=2, default=(64, 64),
                        help="Fingerprint size (width height)")
    parser.add_argument('--block_size', type=int, default=8, help="DCT block size")
    return parser.parse_args()


class AlphaPunchGUI:
    def __init__(self, master, alphapunch):
        self.master = master
        self.alphapunch = alphapunch
        master.title("AlphaPunch Image Fingerprinting")

        self.input_button = tk.Button(master, text="Select Input Image", command=self.select_input)
        self.input_button.pack()

        self.output_button = tk.Button(master, text="Select Output Location", command=self.select_output)
        self.output_button.pack()

        self.embed_button = tk.Button(master, text="Embed Fingerprint", command=self.embed_fingerprint)
        self.embed_button.pack()

        self.verify_button = tk.Button(master, text="Verify Fingerprint", command=self.verify_fingerprint)
        self.verify_button.pack()

    def select_input(self):
        self.input_path = filedialog.askopenfilename()

    def select_output(self):
        self.output_path = filedialog.asksaveasfilename(defaultextension=".png")

    def embed_fingerprint(self):
        if hasattr(self, 'input_path') and hasattr(self, 'output_path'):
            salt, psnr, ssim = self.alphapunch.embed_fingerprint_with_quality(self.input_path, self.output_path)
            messagebox.showinfo("Success", f"Fingerprint embedded successfully!\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        else:
            messagebox.showerror("Error", "Please select input and output paths.")

    def verify_fingerprint(self):
        verify_path = filedialog.askopenfilename()
        if verify_path:
            salt = b''  # In a real scenario, you'd need to store and retrieve the salt
            is_authentic, similarity = self.alphapunch.verify_fingerprint(verify_path, salt)
            result = "Authentic" if is_authentic else "Not authentic"
            messagebox.showinfo("Verification Result", f"Image is {result}\nSimilarity: {similarity:.2%}")


def main():
    args = parse_arguments()
    logger = setup_logger()

    alphapunch = EnhancedAlphaPunch(
        private_key="your_secret_key_here",
        logger=logger,
        fingerprint_size=args.fingerprint_size,
        block_size=args.block_size,
        embed_strength=args.embed_strength
    )

    if args.gui:
        root = tk.Tk()
        AlphaPunchGUI(root, alphapunch)
        root.mainloop()
    elif args.embed:
        if args.input and args.output:
            salt, psnr, ssim = alphapunch.embed_fingerprint_with_quality(args.input, args.output)
            print(f"Fingerprint embedded successfully. PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            print(f"Salt: {salt.hex()}")  # In a real scenario, you'd need to store this securely
        else:
            print("Please provide input and output paths for embedding.")
    elif args.verify:
        if args.input:
            salt = b''  # In a real scenario, you'd need to retrieve the salt
            is_authentic, similarity = alphapunch.verify_fingerprint(args.input, salt)
            print(f"Verification result: {'Authentic' if is_authentic else 'Not authentic'}")
            print(f"Similarity: {similarity:.2%}")
        else:
            print("Please provide an input path for verification.")
    else:
        print("Please specify --embed, --verify, or --gui")


if __name__ == "__main__":
    main()