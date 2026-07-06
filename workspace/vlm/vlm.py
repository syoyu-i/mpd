# qwen_egocentric_describe.py

import os
import glob
import json
import pandas as pd
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_DIR = "./../../../data/images"          # 一人称画像を入れるフォルダ
OUT_CSV = "qwen_egocentric_results.csv"
OUT_JSONL = "qwen_egocentric_results.jsonl"


PROMPT = """
あなたは一人称視点画像を分析するアシスタントです。
画像内で「何が起きているか」を日本語で説明してください。

以下の観点を含めてください。
1. 場所・環境
2. 見えている人物・物体
3. 視点人物が今していそうな行動
4. 周囲で起きている出来事
5. 不確実な点があれば「推測」と明記

出力は簡潔に、3〜5文程度にしてください。
"""


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def describe_image(image_path, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def main():
    model, processor = load_model()

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    image_paths = sorted(image_paths)

    results = []

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i, image_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] {image_path}")

            try:
                description = describe_image(image_path, model, processor)
                row = {
                    "image_path": image_path,
                    "description": description,
                }
            except Exception as e:
                row = {
                    "image_path": image_path,
                    "description": "",
                    "error": str(e),
                }

            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_JSONL}")


if __name__ == "__main__":
    main()