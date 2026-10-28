"""Step 4 - classify new messages with a trained model.

Examples:
  python predict.py --demo
  python predict.py --text "Your account is locked, verify at http://secure-login.example"
  python predict.py --model cnn --file my_messages.txt        (one message per line)
  python predict.py --interactive
"""
import argparse

from config import MODEL_NAMES
from inference import available_models, load, score

DEMO_MESSAGES = [
    ("phishing", "Dear customer, we detected unusual activity on your account. Your access has been "
                 "suspended. Click here http://secure-verify-account.com/login to verify your identity "
                 "within 24 hours or your account will be closed."),
    ("phishing", "Congratulations! You have been selected to receive a $1,000 gift card. Claim your "
                 "prize now by providing your name, address and credit card number for shipping."),
    ("phishing", "URGENT: Your mailbox has exceeded its storage limit. Login at www.mail-upgrade.net "
                 "with your username and password to avoid losing your emails."),
    ("phishing", "I am a lawyer representing the estate of a deceased client who left $15.5 million. "
                 "Kindly reply with your bank details so we can transfer your share of the funds."),
    ("safe", "Hi team, the meeting about the quarterly report has been moved to Thursday at 3pm. "
             "Please bring your notes on the project schedule. Thanks, John"),
    ("safe", "Thanks for sending the draft. I made a few comments on section 2 and attached the "
             "revised version. Let me know if you have questions before Friday."),
    ("safe", "The conference on computational linguistics will be held in June. Papers on syntax, "
             "semantics and morphology are welcome. Submission deadline is March 1."),
    ("safe", "Can you check the gas contract volumes for next month? The deal ticket numbers in the "
             "system don't match what we discussed with the counterparty."),
]


def report(name, texts, truths=None):
    model, vocab = load(name)
    probs, expl = score(name, model, vocab, texts, top_k=6)
    print(f"\nModel: {MODEL_NAMES[name]}\n" + "-" * 70)
    correct = 0
    for i, (text, p) in enumerate(zip(texts, probs)):
        label = "PHISHING" if p >= 0.5 else "safe"
        line = f"[{label:8s}] P(phishing)={p:.3f}"
        if truths:
            ok = (label.lower() == truths[i])
            correct += ok
            line += f"  expected={truths[i]:8s} {'OK' if ok else 'WRONG'}"
        print(line)
        print("   " + (text[:110] + "..." if len(text) > 110 else text))
        if expl:
            print("   key words (attention): " + ", ".join(expl[i]))
    if truths:
        print(f"\n{correct}/{len(texts)} demo messages classified as expected")


def main():
    models = available_models()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=models, default="bilstm" if "bilstm" in models else models[-1])
    parser.add_argument("--text")
    parser.add_argument("--file")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.demo:
        report(args.model, [t for _, t in DEMO_MESSAGES], [lab for lab, _ in DEMO_MESSAGES])
    elif args.text:
        report(args.model, [args.text])
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            report(args.model, [line.strip() for line in f if line.strip()])
    elif args.interactive:
        print("Paste a message and press Enter (empty line to quit).")
        while (msg := input("> ").strip()):
            report(args.model, [msg])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
