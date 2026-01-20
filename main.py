# main.py
import argparse
import torch
import numpy as np
from main_net import DEPTH


# ===== Cipher registry =====
from rectangle import RectangleCipher
from my_speck import SpeckCipher

# ===== Model & training =====
from main_net import train_distinguisher

# ----------------------------
# 1. Cipher registry
# ----------------------------
CIPHER_REGISTRY = {
    "rectangle": RectangleCipher,
    "speck": SpeckCipher,
}

TRAIN_FUNC_REGISTRY = {
    "rectangle": train_distinguisher,
    "speck": train_distinguisher,
}


# ----------------------------
# 2. Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural cryptanalysis framework (multi-cipher)"
    )

    # ---- cipher & data ----
    parser.add_argument("--cipher", type=str, required=True,
                        choices=CIPHER_REGISTRY.keys(),
                        help="cipher algorithm")
    parser.add_argument("--rounds", type=int, default=5,
                        help="number of rounds")
    parser.add_argument("--pairs", type=int, default=8,
                        help="ciphertext pairs per sample")
    parser.add_argument("--data_size", type=int, default=2 ** 18,
                        help="number of samples")

    # ---- training ----
    parser.add_argument("--epochs", type=int, default=20,
                        help="training epochs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)

    # ---- misc ----
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--sanity_random_labels",
        action="store_true",
        help="Sanity check: replace labels with random"
    )

    return parser.parse_args()


# ----------------------------
# 3. Main logic
# ----------------------------
def main():
    args = parse_args()

    # ---- reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")
    print(f"[+] Cipher: {args.cipher}")

    # ---- select cipher ----
    cipher_cls = CIPHER_REGISTRY[args.cipher]
    cipher = cipher_cls()

    # ---- set model parameters dynamically ----
    if args.cipher == "rectangle":
        num_blocks = 6  # RECTANGLE 每 ciphertext pair 包含 6 blocks
    elif args.cipher == "speck":
        num_blocks = 6  # SPECK 每 ciphertext pair 包含 2 words

    # ---- call training ----
    model, history = train_distinguisher(
        cipher=cipher,  # 传入 cipher 对象
        num_epochs=args.epochs,  # 训练轮数
        num_rounds=args.rounds,  # 加密轮数
        pairs=args.pairs,  # 每条样本的密文对数量
        depth=getattr(args, "depth", DEPTH),  # 可选命令行参数 depth，否则使用默认
        num_blocks=num_blocks
    )


# ----------------------------
# 4. Entry point
# ----------------------------
if __name__ == "__main__":
    main()
