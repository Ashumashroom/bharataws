from transformers import MarianMTModel, MarianTokenizer
import warnings

warnings.filterwarnings("ignore")

class TranslatorEngine:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-es"):
        print(f"Loading Translation Model ({model_name})...")
        # Explicitly loading the tokenizer and model prevents pipeline registry errors
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        """Translates text to target language."""
        print("🧠 Translating via Hugging Face...")
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate the translation
        translated_tokens = self.model.generate(**inputs)
        
        # Decode the tokens back into a human-readable string
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        print(f"🌐 Translated Text: '{translated_text}'")
        return translated_text