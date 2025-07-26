import os
os.environ["STABILITY_API_KEY"] = "sk-IAO1ufNNnTY2k1sjMnMhkMtL75q1tTA6cLMPbnuF9FhMDGSl"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBtlSTfAunr6e2vXJKUCwKAJLrgSledzc4"
import io
import requests
import google.generativeai as genai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')

# Initialize models
text_model = genai.GenerativeModel('gemini-1.5-flash')
vision_model = genai.GenerativeModel('gemini-1.5-flash')

def analyze_image(image, prompt="Describe this image in detail"):
    """Analyze uploaded image using Gemini"""
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()
        
        # Generate content
        response = vision_model.generate_content(
            [prompt, {"mime_type": "image/png", "data": img_data}],
            stream=True
        )
        response.resolve()
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def generate_image_from_text(prompt):
    """Generate image using Stability AI"""
    try:
        if not STABILITY_API_KEY:
            raise Exception("Missing Stability API key")

        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30
            }
        )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")

        data = response.json()
        import base64
        img_data = base64.b64decode(data["artifacts"][0]["base64"])
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        return None

def main():
    st.title("Multi-Modal Chatbot")
    
    # Create tabs for different functions
    tab1, tab2 = st.tabs(["🖼️ Image Analysis", "📝 Text Analysis"])
    
    with tab1:
        st.header("Analyze Uploaded Images")
        uploaded_file = st.file_uploader("Choose an image...", 
                                      type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            prompt = st.text_input("Ask about the image (optional):", 
                                 "Describe this image in detail")
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    description = analyze_image(image, prompt)
                    st.subheader("Analysis Results")
                    st.write(description)
    
    with tab2:
        st.header("Generate Images from Text")
        text_prompt = st.text_area("Describe what you want to generate:", 
                                 "A realistic photo of a dragon flying over mountains")
        
        if st.button("Generate Image"):
            with st.spinner("Creating your image..."):
                generated_image = generate_image_from_text(text_prompt)
                if generated_image:
                    st.image(generated_image, 
                           caption=text_prompt, 
                           use_container_width=True)
                    st.success("Image generated successfully!")
                else:
                    st.error("Failed to generate image")

if __name__ == "__main__":
    # Verify environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("Missing Google API key in .env file")
    if not os.getenv('STABILITY_API_KEY'):
        st.error("Missing Stability API key in .env file")
    
    main()