import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import pyautogui

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def capture_screen():
    screenshot = pyautogui.screenshot()
    screenshot.save("screen.png")
    return "screen.png"

def analyze_screen():
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        image_path = capture_screen()
        image = Image.open(image_path)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                image,
                "Analyze this screen in simple words. If there is code or an error, explain it clearly."
            ]
        )

        return response.text

    except Exception as e:
        return f"AI Screen Analyzer unavailable due to Gemini API quota or setup issue: {e}"

if __name__ == "__main__":
    result = analyze_screen()
    print(result)