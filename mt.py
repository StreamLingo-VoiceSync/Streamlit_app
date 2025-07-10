# mt_2.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect

# Load NLLB model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Map language codes to NLLB codes
lang_map = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans"
}

def translate_text(text, target_lang_code):
    try:
        # Step 1: Detect source language using langdetect
        detected_lang = detect(text)  # returns ISO code like 'en', 'hi', etc.

        # Step 2: Map to NLLB lang codes
        src_lang = lang_map.get(detected_lang)
        tgt_lang = lang_map.get(target_lang_code)

        if not src_lang or not tgt_lang:
            return f"[ERROR] Language not supported. Detected: {detected_lang}, Target: {target_lang_code}"

        # Step 3: Run NLLB translation
        translator = pipeline("translation", model=model, tokenizer=tokenizer,
                              src_lang=src_lang, tgt_lang=tgt_lang)

        result = translator(text, max_length=400)
        return result[0]['translation_text']

    except Exception as e:
        return f"[ERROR] {str(e)}"
