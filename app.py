# --- 1. IMPORTS ---
import os
import json
import numpy as np
import markdown
import time
from PIL import Image

# Import the Hugging Face Inference Client
from huggingface_hub import InferenceClient

import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr

print("--- Libraries imported successfully ---")

# --- 2. CONFIGURATION AND INITIALIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_FILE_PATH = "59skinalyze_model_best.h5"
class_names = [
    'Acne', 'Actinic keratosis', 'Actinic_Cheilitis', 'Athlete_Foot', 'Atopic_Dermatitis', 'Ba_Cellulitis', 'Ba_Impetigo',
    'Basal_Cell_Carcinoma_and_Other_Carcinoma', 'Benign_Keratosis', 'Benign_Tumors', 'Black Heel & Corn', 'Bullous_Disease_Photos',
    'Cellulitis_Impetigo_And_Other_Bacterial_Infections', 'Chickenpox', 'Cutaenous_T-cell_Lymphoma', 'Dermatitis',
    'Dermatitis_Herpetiformis', 'Dermatofibroma', 'Dry_Skin', 'Eczema', 'Exanthems_And_Drug_Eruptions',
    'Hair_Loss_Photos_Alopecia_And_Other_Hair_Diseases', 'Herpes', 'Hidradenitis-Suppurativa', 'Hpv_And_Other_Stds_Photos',
    'Ichthiosis', 'Keratosis_Pilaris', 'Light_Diseases_And_Disorders_Of_Pigmentation', 'Lupus_And_Other_Connective_Tissue_Diseases',
    'Malignant_Lesions', 'Malignant_Tumors', 'Melanoma_Skin_Cancer_Nevi_And_Moles', 'Moles', 'Molluscum',
    'Nail Fungus And Other Nail Disease', 'Normal_Skin', 'Oily_Skin', 'Other diseases', 'Pa_Cutaneous_Larva_Migrans',
    'Poison_Ivy_Photos_And_Other_Contact_Dermatitis', 'Psoriasis_Pictures_Lichen_Planus_And_Related_Diseases', 'Rashes',
    'Ringworm', 'Rosacea', 'Scabies_Lyme_Disease_And_Other_Infestations_And_Bites',
    'Seborrheic_Keratoses_And_Other_Benign_Tumors', 'Shingles', 'Sun_Sunlight_Damage', 'Systemic_Disease',
    'Tinea_Ringworm_Candidiasis_And_Other_Fungal_Infections', 'Urticaria_Hives', 'Vascular_Lesion', 'Vascular_Tumors',
    'Vasculitis_Photos', 'Vitigo', 'Warts'
]

# --- Configure the Hugging Face Inference Client ---
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("Hugging Face Token loaded successfully.")
    # We will use a powerful and popular instruction-tuned model from Mistral AI
    inference_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)
else:
    print("WARNING: Could not load Hugging Face Token. Detailed info will be disabled.")
    inference_client = None

# --- 3. DEFINE CUSTOM MODEL ARCHITECTURE COMPONENTS ---
def squeeze_excite_block(input_tensor, ratio=8):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Conv2D(channels // ratio, (1, 1), activation='relu', padding='same')(se)
    se = layers.Conv2D(channels, (1, 1), activation='sigmoid', padding='same')(se)
    return layers.Multiply()([input_tensor, se])

# --- 4. LOAD THE TRAINED MODEL ---
print("--- Loading the trained Skinalyze model ---")
IMG_HEIGHT, IMG_WIDTH = 180, 180
skinalyze_model = None
try:
    custom_objects = { "squeeze_excite_block": squeeze_excite_block, "LeakyReLU": layers.LeakyReLU }
    skinalyze_model = models.load_model(MODEL_FILE_PATH, custom_objects=custom_objects, compile=False)
    print("Trained model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Could not load model. Error: {e}")

# --- 5. DEFINE THE CORE INFERENCE FUNCTION ---
DANGER_LEVELS = { "level_1": {"text": "Treatable", "color": "#22c55e"}, "level_2": {"text": "Visit a Doctor", "color": "#f59e0b"}, "level_3": {"text": "Specialist Needed", "color": "#ef4444"}, "level_4": {"text": "Urgent Attention Recommended", "color": "#8b0000"} }
CONDITION_SEVERITY_MAP = { 'Acne': "level_1", 'Athlete_Foot': "level_1", 'Benign_Keratosis': "level_1", 'Black Heel & Corn': "level_1", 'Dry_Skin': "level_1", 'Keratosis_Pilaris': "level_1", 'Nail Fungus And Other Nail Disease': "level_1", 'Normal_Skin': "level_1", 'Oily_Skin': "level_1", 'Poison_Ivy_Photos_And_Other_Contact_Dermatitis': "level_1", 'Ringworm': "level_1", 'Seborrheic_Keratoses_And_Other_Benign_Tumors': "level_1", 'Tinea_Ringworm_Candidiasis_And_Other_Fungal_Infections': "level_1", 'Warts': "level_1", 'Atopic_Dermatitis': "level_2", 'Ba_Impetigo': "level_2", 'Benign_Tumors': "level_2", 'Chickenpox': "level_2", 'Dermatitis': "level_2", 'Dermatofibroma': "level_2", 'Eczema': "level_2", 'Exanthems_And_Drug_Eruptions': "level_2", 'Hair_Loss_Photos_Alopecia_And_Other_Hair_Diseases': "level_2", 'Herpes': "level_2", 'Hpv_And_Other_Stds_Photos': "level_2", 'Light_Diseases_And_Disorders_Of_Pigmentation': "level_2", 'Moles': "level_2", 'Molluscum': "level_2", 'Other diseases': "level_2", 'Pa_Cutaneous_Larva_Migrans': "level_2", 'Psoriasis_Pictures_Lichen_Planus_And_Related_Diseases': "level_2", 'Rashes': "level_2", 'Rosacea': "level_2", 'Scabies_Lyme_Disease_And_Other_Infestations_And_Bites': "level_2", 'Sun_Sunlight_Damage': "level_2", 'Urticaria_Hives': "level_2", 'Vascular_Lesion': "level_2", 'Vitigo': "level_2", 'Actinic keratosis': "level_3", 'Actinic_Cheilitis': "level_3", 'Basal_Cell_Carcinoma_and_Other_Carcinoma': "level_3", 'Bullous_Disease_Photos': "level_3", 'Cutaenous_T-cell_Lymphoma': "level_3", 'Dermatitis_Herpetiformis': "level_3", 'Hidradenitis-Suppurativa': "level_3", 'Ichthiosis': "level_3", 'Lupus_And_Other_Connective_Tissue_Diseases': "level_3", 'Malignant_Lesions': "level_3", 'Malignant_Tumors': "level_3", 'Melanoma_Skin_Cancer_Nevi_And_Moles': "level_3", 'Vascular_Tumors': "level_3", 'Vasculitis_Photos': "level_3", 'Ba_Cellulitis': "level_4", 'Cellulitis_Impetigo_And_Other_Bacterial_Infections': "level_4", 'Shingles': "level_4", 'Systemic_Disease': "level_4" }

def classify_and_get_info(image):
    if skinalyze_model is None: return "<p style='color: red; font-weight: bold;'>Error: Model not loaded.</p>", gr.update(visible=True)
    if image is None: return "<p style='color: orange;'>Please upload an image first.</p>", gr.update(visible=True)
    try:
        img_array = np.array(image.resize((IMG_HEIGHT, IMG_WIDTH)))
        img_array = np.expand_dims(img_array, axis=0)
        predictions = skinalyze_model.predict(img_array, verbose=0)
        confidence = np.max(predictions[0])
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_key = class_names[predicted_class_index]
        predicted_class_name = predicted_class_key.replace('_', ' ')
        level_key = CONDITION_SEVERITY_MAP.get(predicted_class_key, "level_2")
        danger_level = DANGER_LEVELS[level_key]
        danger_level_html = f"""<div style='margin-top: 20px;'><h3 style='color: #000000; margin-bottom: 5px;'>Severity Assessment</h3><div style='padding: 15px; border-radius: 8px; background-color: {danger_level['color']}; text-align: center;'><p style='font-size: 1.4em; color: white; font-weight: bold; margin: 0;'>{danger_level['text']}</p></div></div>"""
        analysis_result = f"""<div style='padding: 20px; border-radius: 10px; background-color: #e0e7ff; font-family: sans-serif; border: 1px solid #c7d2fe;'><h2 style='color: #000000; margin-top: 0;'>AI Analysis Complete</h2><p style='font-size: 1.1em; color: #000000;'><strong>Preliminary Finding:</strong> {predicted_class_name}</p><p style='font-size: 1.1em; color: #000000;'><strong>Confidence Level:</strong> {confidence:.2%}</p></div>{danger_level_html}"""
        
        if not inference_client:
            return analysis_result + "<p style='color: orange; margin-top: 15px;'><i>Detailed information is unavailable.</i></p>", gr.update(visible=True)
        
        # This special formatting tells the Mistral model how to behave.
        system_prompt = "You are a medical information assistant. Provide a clear, structured summary for the given skin condition using Markdown format exactly as follows, without a main title: ### **Condition Overview** * Description. ### **Common Symptoms** * Symptom 1 * Symptom 2. ### **General Recommendations** * Advice 1. ### **When to Consult a Doctor** * A clear statement advising consultation."
        user_prompt = f"Provide a summary for the skin condition: \"{predicted_class_name}\"."
        
        # --- THIS IS THE NEW HUGGING FACE API CALL ---
        try:
            # We use the chat_completion endpoint which is great for instruction-following models
            response = inference_client.chat_completion(
                messages=[
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                ],
                max_tokens=500,
            )
            hf_text = response.choices[0].message.content
        except Exception as e:
            print(f"Hugging Face API call failed. Error: {e}")
            hf_text = None
        # --------------------------------------------------------
        
        if hf_text:
            hf_html = markdown.markdown(hf_text)
            detailed_info = f"""<div style='margin-top: 20px; padding: 20px; border-radius: 10px; background-color: #f9fafb; font-family: sans-serif; border: 1px solid #e5e7eb;'><h2 style='color: #000000; margin-top: 0;'>Detailed Information</h2><div style='color: #000000;'>{hf_html}</div></div>"""
        else:
            detailed_info = "<p style='color: red; margin-top: 15px;'>Could not retrieve detailed information from the API at this time.</p>"
        
        return analysis_result + detailed_info, gr.update(visible=True)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return f"<p style='color: red; font-weight: bold;'>An unexpected error occurred: {e}</p>", gr.update(visible=True)

# --- 6. DEFINE AND LAUNCH THE GRADIO INTERFACE ---
with gr.Blocks(css="body {font-family: 'sans-serif';}") as demo:
    gr.Markdown("# 🔬 Skinalyze AI")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Skin Image for Analysis")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column(scale=2):
            with gr.Column(visible=False) as results_col:
                results_output = gr.HTML()
    gr.Markdown("⚠️ Disclaimer: This is an educational tool, not a substitute for professional medical advice.")
    def clear_results_and_hide(): return "", gr.update(visible=False)
    submit_btn.click(fn=classify_and_get_info, inputs=input_image, outputs=[results_output, results_col])
    input_image.clear(fn=clear_results_and_hide, outputs=[results_output, results_col])
    input_image.upload(fn=clear_results_and_hide, outputs=[results_output, results_col])

if __name__ == "__main__":
    demo.launch()