#!/usr/bin/env python3
"""
Batch-generate stylized actor portraits with OpenAI's GPT Image API.

Modes:
- TEXT-ONLY (no --style): uses images.generate (quality supported).
- STYLE-REF  (--style):   uses images.edit to guide style from an image (quality ignored).

Usage examples (PowerShell):
  python batch_portraits_v2.py --csv actors.csv --size 1024 --format png --quality medium
  python batch_portraits_v2.py --csv actors.csv --style Keanu_Reeves-Chinisorie-cropped.png --size 1024 --format png

Requirements:
  pip install openai python-dotenv
  setx OPENAI_API_KEY "sk-..."
"""
import argparse
import base64
import csv
import io
import time
from pathlib import Path
from openai import OpenAI

def make_prompt(person_name: str) -> str:
    # You can tweak this to taste.
    return f"""
Create a clean, blue-ink portrait in the exact style of the provided reference image:
- single-color ink drawing and bold contour lines
- clean white background
- Slightly Imperfect, Human
Subject: {person_name}.
Clothes: Usual Clothing associated with this person.  Do NOT copy the clothes from the Referance Photo. 
Expression: Usual expression associated with this persons charater. keep the face CLEAN from marks.
Bonus points (Extra Money): if the person is an Actor for tv series or film.  then draw them from there most popular show / film.
Framing: FACE ONLY! NO SHOULDERS, Tight crop on the face, centered. 1:1 square composition with generous WHITE SPACE ABOVE THE HEAD.
Important: Use the first input image only as a STYLE REFERENCE. Do NOT copy that person's identity or their features or clothes.
minimal shading on face.  
"""
# avoid shading that looks photographic.
# Shoulders / Bust

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with a 'name' column")
    parser.add_argument("--style", help="Path to a style reference image (PNG/JPG). If set, uses image-to-image (edit).")
    parser.add_argument("--out", default="out", help="Output folder")
    parser.add_argument("--size", type=int, default=1024, choices=[1024, 1536, 2048], help="Square image size in px")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests")
    parser.add_argument("--format", default="png", choices=["png", "jpeg", "webp"], help="Output format")
    parser.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"], help="(generate mode only) quality hint")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()  # reads OPENAI_API_KEY from env
    size_str = f"{args.size}x{args.size}"

    # Load CSV
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "name" not in reader.fieldnames:
            raise SystemExit("CSV must have a header column named 'name'")
        rows = [r for r in reader if r.get("name", "").strip()]

    # Load the style image bytes once (if provided)
    style_bytes = None
    if args.style:
        with open(args.style, "rb") as sf:
            style_bytes = sf.read()
        print("Mode: STYLE-REF (images.edit). Quality flag will be ignored.")
    else:
        print("Mode: TEXT-ONLY (images.generate). Will use --quality =", args.quality)

    for i, row in enumerate(rows, 1):
        person_name = row["name"].strip()
        prompt = make_prompt(person_name)
        out_path = out_dir / f"{person_name.replace(' ', '_')}.{args.format.lower()}"
        print(f"[{i}/{len(rows)}] {person_name} -> {out_path.name}")

        try:
            if style_bytes:
                bio = io.BytesIO(style_bytes)
                bio.name = Path(args.style).name
                result = client.images.edit(
                    model="gpt-image-1",
                    image=bio,
                    prompt=prompt,
                    size=size_str,
                    output_format=args.format.lower(),
                )
            else:
                try:
                    result = client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt,
                        size=size_str,
                        quality=args.quality,
                        output_format=args.format.lower(),
                    )
                except TypeError:
                    result = client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt,
                        size=size_str,
                        output_format=args.format.lower(),
                    )

            b64 = result.data[0].b64_json
            with open(out_path, "wb") as img_f:
                img_f.write(base64.b64decode(b64))

        except Exception as e:
            print(f"!! Error for {person_name}: {e}")
        finally:
            time.sleep(args.sleep)

    print("Done. Files saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
