import os
import requests
import cv2
import numpy as np
from flask import Flask, request, send_from_directory, jsonify
from dotenv import load_dotenv
import base64
import time
import jwt
import logging
import random
import json
from typing import Optional, Dict, Any, Union, Tuple
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Get the directory containing the script
BASE_DIR = Path(__file__).resolve().parent

# Delete any existing .env files first
def cleanup_env_files():
    env_files = [
        BASE_DIR / '.env',
        BASE_DIR / 'whatsapp-tryon-bot' / '.env'
    ]
    for env_file in env_files:
        if env_file.exists():
            print(f"Removing {env_file}")
            env_file.unlink()

# Create new .env file with correct values
def create_env_file():
    env_content = """WHATSAPP_API_VERSION=v17.0
PHONE_NUMBER_ID=607253892466255
ACCESS_TOKEN=EAAdtq4qJH50BOZBvWMs9gF92PzZARYKPPPO3xiyy5Qj5TgLrm5ZA5xD10x28wZBexPMrdrZBiZCUTAjCU9x07hV1wpFFypjdAU30IiccM7ZBxa7ZAKmqhNFZB3oNfCK3SaIdNNvbaE2JKDuZCAeuuaBVKiFXeSyNEBGOJBqUHhpwFrpKmZBwBgylW7x6tQNfS2ZBMgZDZD
IMAGE_URL=https://6b83-103-134-1-39.ngrok-free.app
WEBSITE_URL=https://fashioncore.com/tryon
VERIFY_TOKEN=1122"""
    
    env_path = BASE_DIR / '.env'
    env_path.write_text(env_content)
    print(f"Created new .env at: {env_path}")

# Clean up and create new .env
cleanup_env_files()
create_env_file()

# Load the environment variables
env_path = BASE_DIR / '.env'
load_dotenv(env_path, override=True)

# Verify loaded values
print("\nLoaded environment variables:")
for var in ['VERIFY_TOKEN', 'WHATSAPP_API_VERSION', 'PHONE_NUMBER_ID', 'IMAGE_URL', 'WEBSITE_URL']:
    print(f"{var}: {os.getenv(var)}")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure static file serving
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Constants
MAX_SEED = 999999

# Brand name constants
BRAND_NAME = "FashionCore Magic Try-on"
BOT_NAME = "FashionCore Assistant"

# Add these state constants
class UserState:
    IDLE = "idle"
    WAITING_FOR_PERSON = "waiting_for_person"
    WAITING_FOR_GARMENT = "waiting_for_garment"
    PROCESSING = "processing"
    SHOWING_RESULT = "showing_result"

# Add these global dictionaries
user_states = {}  # Format: {phone_number: UserState}
user_images = {}  # Format: {phone_number: {'person': image_path, 'garment': image_path}}
user_results = {}  # Format: {phone_number: {'result_url': url}}

class KlingAIClient:
    def __init__(self):
        self.access_key = "b8afed7981f24e29855104ddf9d6df5f"
        self.secret_key = "4543919766a74bb3bbcbb937dd252ff1"
        self.base_url = "https://api.klingai.com"
        self.logger = logging.getLogger(__name__)
    
    def _generate_jwt_token(self) -> str:
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        }
        return jwt.encode(payload, self.secret_key, headers=headers)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self._generate_jwt_token()}"
        }
    
    def try_on(self, person_img: np.ndarray, garment_img: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
        """
        Use the Kling AI's Virtual Try-on API to generate a try-on image.
        
        Args:
            person_img: The person's image
            garment_img: The garment image
            seed: Random seed for generation
            
        Returns:
            The resulting image and status message
        """
        if person_img is None or garment_img is None:
            raise ValueError("Empty image")
            
        # Encode images
        encoded_person = cv2.imencode('.jpg', cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))[1]
        encoded_person = base64.b64encode(encoded_person.tobytes()).decode('utf-8')
        
        encoded_garment = cv2.imencode('.jpg', cv2.cvtColor(garment_img, cv2.COLOR_RGB2BGR))[1]
        encoded_garment = base64.b64encode(encoded_garment.tobytes()).decode('utf-8')

        # Submit task using the improved V1.5 model
        url = f"{self.base_url}/v1/images/kolors-virtual-try-on"
        data = {
            "model_name": "kolors-virtual-try-on-v1-5",  # Using the improved V1.5 model
            "cloth_image": encoded_garment,
            "human_image": encoded_person,
            "seed": seed
        }

        try:
            self.logger.info("Making API request to Kling AI Virtual Try-on service")
            response = requests.post(
                url, 
                headers=self._get_headers(),
                json=data,
                timeout=50
            )
            
            if response.status_code == 429:
                error_msg = "Sorry, our service is currently at capacity. Please try again in a few minutes."
                self.logger.error(f"API rate limit exceeded: {response.text}")
                return None, error_msg
                
            if response.status_code != 200:
                error_msg = f"Error: API returned status code {response.status_code}"
                self.logger.error(f"API error: {response.text}")
                return None, error_msg
            
            result = response.json()
            task_id = result['data']['task_id']
            
            # Wait for result
            self.logger.info(f"Task submitted successfully. Task ID: {task_id}")
            self.logger.info("Waiting for try-on result...")
            
            # Initial wait
            time.sleep(9)
            
            for attempt in range(12):
                try:
                    url = f"{self.base_url}/v1/images/kolors-virtual-try-on/{task_id}"
                    response = requests.get(url, headers=self._get_headers(), timeout=20)
                    
                    if response.status_code != 200:
                        self.logger.error(f"Error checking task status: {response.text}")
                        time.sleep(1)
                        continue
                    
                    result = response.json()
                    status = result['data']['task_status']
                    
                    if status == "succeed":
                        output_url = result['data']['task_result']['images'][0]['url']
                        self.logger.info(f"Try-on successful! Result URL: {output_url}")
                        
                        img_response = requests.get(output_url)
                        img_response.raise_for_status()
                        
                        nparr = np.frombuffer(img_response.content, np.uint8)
                        result_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        return result_img, "Success"
                    elif status == "failed":
                        error_msg = f"Sorry, we couldn't create the try-on image. {result['data']['task_status_msg']}"
                        self.logger.error(f"Task failed: {result['data']['task_status_msg']}")
                        return None, error_msg
                    else:
                        self.logger.info(f"Task status: {status}. Waiting...")
                        
                except requests.exceptions.ReadTimeout:
                    self.logger.warning(f"Timeout on attempt {attempt+1}/12. Retrying...")
                    if attempt == 11:
                        return None, "Sorry, the try-on is taking longer than expected. Please try again."
                        
                time.sleep(1)
                
            return None, "The try-on is taking too long. Please try again later."
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Sorry, we're having trouble connecting to our service. Please try again later."
            self.logger.error(f"API error: {str(e)}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Sorry, something went wrong. Please try again later."
            self.logger.error(f"Unexpected error: {str(e)}")
            return None, error_msg

def send_whatsapp_message(to: str, message: str):
    """Send a text message via WhatsApp"""
    url = f"https://graph.facebook.com/{os.getenv('WHATSAPP_API_VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"
    
    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info(f"Message sent successfully to {to}")
        logger.debug(f"Response: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        if hasattr(e, 'response'):
            logger.error(f"Response content: {e.response.text}")
        return False

def send_whatsapp_image(to_number: str, image_url: str, caption: str = ""):
    """Send an image message via WhatsApp"""
    try:
        url = f"https://graph.facebook.com/{os.getenv('WHATSAPP_API_VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "image",
            "image": {
                "link": image_url,
                "caption": caption
            }
        }
        
        headers = {
            "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Sending image to {to_number} with URL: {image_url}")
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        logger.info(f"Image send response: {json.dumps(response_data)}")
        
        if response.status_code != 200:
            logger.error(f"Failed to send image. Status: {response.status_code}, Response: {response_data}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error sending WhatsApp image: {str(e)}", exc_info=True)
        return False

def send_whatsapp_interactive_message(to_number: str, message: str, buttons: list):
    """Send an interactive message with buttons via WhatsApp"""
    try:
        url = f"https://graph.facebook.com/{os.getenv('WHATSAPP_API_VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {
                    "text": message
                },
                "action": {
                    "buttons": buttons
                }
            }
        }
        
        headers = {
            "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Sending interactive message to {to_number}")
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        logger.info(f"Interactive message response: {json.dumps(response_data)}")
        
        if response.status_code != 200:
            logger.error(f"Failed to send interactive message. Status: {response.status_code}, Response: {response_data}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error sending WhatsApp interactive message: {str(e)}", exc_info=True)
        return False

def download_whatsapp_image(image_id: str) -> Optional[str]:
    """Download image from WhatsApp"""
    logger.info(f"Starting download for image ID: {image_id}")
    
    try:
        url = f"https://graph.facebook.com/{os.getenv('WHATSAPP_API_VERSION')}/{image_id}"
        headers = {"Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}"}
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to get image URL: {response.text}")
            return None
            
        image_data = response.json()
        if 'url' not in image_data:
            logger.error("No URL in image metadata")
            return None
            
        image_response = requests.get(image_data['url'], headers=headers)
        if image_response.status_code != 200:
            logger.error(f"Failed to download image: {image_response.status_code}")
            return None
            
        filename = f"image_{image_id}.jpg"
        with open(filename, 'wb') as f:
            f.write(image_response.content)
            
        # Verify image can be opened
        test_img = cv2.imread(filename)
        if test_img is None:
            logger.error("Downloaded image cannot be opened")
            return None
            
        logger.info(f"Successfully downloaded and verified image: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return None

def generate_unique_filename():
    """Generate a unique filename using timestamp and UUID"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    return f"result_{timestamp}_{unique_id}.png"

def process_images(person_image_path: str, garment_image_path: str) -> Tuple[Optional[str], str]:
    """Process images with the virtual try-on service"""
    try:
        logger.info(f"Processing images: {person_image_path} and {garment_image_path}")
        
        # Load person image
        logger.info("Loading person image")
        person_img = cv2.imread(person_image_path)
        if person_img is None:
            logger.error("Failed to load person image")
            return None, "We couldn't process your photo. Please ensure it's clearly visible and try again."
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        logger.info(f"Person image loaded successfully. Shape: {person_img.shape}")
        
        # Load garment image
        logger.info("Loading garment image")
        garment_img = cv2.imread(garment_image_path)
        if garment_img is None:
            logger.error("Failed to load garment image")
            return None, "We couldn't process the garment image. Please ensure it has a clear view of the clothing and try again."
        garment_img = cv2.cvtColor(garment_img, cv2.COLOR_BGR2RGB)
        logger.info(f"Garment image loaded successfully. Shape: {garment_img.shape}")
        
        # Initialize client
        logger.info("Initializing FashionCore Magic Try-on client")
        client = KlingAIClient()
        
        # Process images
        logger.info("Calling FashionCore Magic Try-on service")
        result_img, status_message = client.try_on(person_img, garment_img, random.randint(0, MAX_SEED))
        
        if result_img is None:
            logger.error(f"FashionCore processing failed. Status: {status_message}")
            return None, status_message
            
        # Save result - FIX: Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        result_filename = generate_unique_filename()
        result_path = os.path.join('static', result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        # FIX: Correct URL construction
        base_url = os.getenv('IMAGE_URL', 'https://6b83-103-134-1-39.ngrok-free.app')
        # Remove any trailing slash from base_url
        base_url = base_url.rstrip('/')
        
        return f"{base_url}/static/{result_filename}", "Success"
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        return None, "Sorry, something went wrong while generating your try-on. Please try again later."

def handle_message(message: dict, sender_number: str):
    """Handle incoming messages based on user state"""
    try:
        current_state = user_states.get(sender_number, UserState.IDLE)
        message_type = message.get('type')
        logger.info(f"Handling message from {sender_number}. Type: {message_type}, State: {current_state}")

        if message_type == 'text':
            text = message.get('text', {}).get('body', '').lower()
            
            # Handle interactive message responses
            if message.get('interactive'):
                button_reply = message.get('interactive', {}).get('button_reply', {})
                if button_reply:
                    button_id = button_reply.get('id')
                    if button_id == 'video_yes' and current_state == UserState.SHOWING_RESULT:
                        # User wants to try video feature
                        website_url = os.getenv('WEBSITE_URL', 'https://fashioncore.com/tryon')
                        send_whatsapp_message(
                            sender_number, 
                            f"Great! You can create amazing try-on videos at our website:\n\n{website_url}\n\nSend 'start' anytime to try on another outfit with FashionCore Magic!"
                        )
                        # Reset state
                        user_states[sender_number] = UserState.IDLE
                        return
                    elif button_id == 'video_no' and current_state == UserState.SHOWING_RESULT:
                        # User doesn't want to try video
                        send_whatsapp_message(
                            sender_number, 
                            "No problem! Send 'start' anytime to try on another outfit with FashionCore Magic!"
                        )
                        # Reset state
                        user_states[sender_number] = UserState.IDLE
                        return
            
            # Regular text message handling
            if text == 'start':
                user_states[sender_number] = UserState.WAITING_FOR_PERSON
                send_whatsapp_message(
                    sender_number, 
                    f"👋 Welcome to {BRAND_NAME}! Let's create a stunning virtual outfit for you.\n\nFirst, please send a full-body photo of yourself standing straight against a plain background."
                )
            elif current_state == UserState.SHOWING_RESULT:
                # Any text in showing_result state should prompt asking if they want to try again
                send_whatsapp_message(
                    sender_number, 
                    "Would you like to try on another outfit? Send 'start' to begin again!"
                )
            else:
                send_whatsapp_message(
                    sender_number, 
                    f"👋 Welcome to {BRAND_NAME}! Send 'start' to begin the virtual try-on experience."
                )
                
        elif message_type == 'image':
            if current_state == UserState.WAITING_FOR_PERSON:
                # Handle first image (person)
                image_id = message.get('image', {}).get('id')
                if not image_id:
                    send_whatsapp_message(
                        sender_number, 
                        "I couldn't process that image. Please try sending it again."
                    )
                    return
                    
                image_path = download_whatsapp_image(image_id)
                if image_path:
                    user_images[sender_number] = {'person': image_path}
                    user_states[sender_number] = UserState.WAITING_FOR_GARMENT
                    send_whatsapp_message(
                        sender_number, 
                        "Perfect! Now please send an image of the clothing item you want to try on. For best results, the garment should be shown on a white background."
                    )
                else:
                    send_whatsapp_message(
                        sender_number, 
                        "I'm having trouble downloading your image. Please try sending a different photo."
                    )
                    
            elif current_state == UserState.WAITING_FOR_GARMENT:
                # Handle second image (garment)
                image_id = message.get('image', {}).get('id')
                if not image_id:
                    send_whatsapp_message(
                        sender_number, 
                        "I couldn't process that image. Please try sending it again."
                    )
                    return
                    
                image_path = download_whatsapp_image(image_id)
                if image_path:
                    # Let the user know we're processing
                    send_whatsapp_message(
                        sender_number, 
                        f"✨ Creating your outfit with {BRAND_NAME} magic! This should take about 15-20 seconds..."
                    )
                    
                    user_states[sender_number] = UserState.PROCESSING
                    
                    try:
                        # Process the images
                        result_url, status_message = process_images(
                            user_images[sender_number]['person'],
                            image_path
                        )
                        
                        # Handle success or failure
                        if result_url:
                            # Save the result URL for later reference
                            user_results[sender_number] = {'result_url': result_url}
                            
                            # Test the result URL before sending to WhatsApp
                            try:
                                # FIX: Test the URL accessibility
                                test_response = requests.head(result_url, timeout=5)
                                test_response.raise_for_status()
                                
                                # Send the result image
                                success = send_whatsapp_image(
                                    sender_number, 
                                    result_url, 
                                    f"✨ Here's your {BRAND_NAME} result! What do you think?"
                                )
                                
                                if success:
                                    # Update user state
                                    user_states[sender_number] = UserState.SHOWING_RESULT
                                    
                                    # Ask if they want to try video
                                    time.sleep(2)  # Brief pause before sending follow-up
                                    buttons = [
                                        {
                                            "type": "reply",
                                            "reply": {
                                                "id": "video_yes",
                                                "title": "Yes, please!"
                                            }
                                        },
                                        {
                                            "type": "reply",
                                            "reply": {
                                                "id": "video_no",
                                                "title": "No, thanks"
                                            }
                                        }
                                    ]
                                    send_whatsapp_interactive_message(
                                        sender_number,
                                        "Would you like to see how this outfit looks in motion? We can create a video try-on too! Try our website 'https://fashioncore.onrender.com/' !!!",
                                        buttons
                                    )
                                else:
                                    # FIX: Fallback if image sending fails - send a text instead
                                    send_whatsapp_message(
                                        sender_number, 
                                        f"I created your try-on image! You can view it at: {result_url}\n\nWould you like to create more outfits? Send 'start' to try again."
                                    )
                                    user_states[sender_number] = UserState.IDLE
                            except requests.RequestException as re:
                                logger.error(f"Failed to access result URL: {re}")
                                send_whatsapp_message(
                                    sender_number, 
                                    "I created your try-on but had trouble sending the image. Please try again by sending 'start'."
                                )
                                user_states[sender_number] = UserState.IDLE
                        else:
                            # Handle processing failure
                            send_whatsapp_message(
                                sender_number, 
                                f"Sorry, {status_message} Please try again by sending 'start'."
                            )
                            user_states[sender_number] = UserState.IDLE
                    except Exception as e:
                        logger.error(f"Error in processing: {str(e)}", exc_info=True)
                        send_whatsapp_message(
                            sender_number, 
                            "Sorry, something went wrong while creating your try-on. Please try again by sending 'start'."
                        )
                        user_states[sender_number] = UserState.IDLE
                    finally:
                        # Clean up
                        if sender_number in user_images:
                            try:
                                if 'person' in user_images[sender_number]:
                                    os.remove(user_images[sender_number]['person'])
                                os.remove(image_path)
                            except Exception as e:
                                logger.error(f"Error removing temporary files: {str(e)}")
                            # Keep the user_images entry for reference
                else:
                    send_whatsapp_message(
                        sender_number, 
                        "I'm having trouble downloading the garment image. Please try sending a different photo."
                    )
            else:
                send_whatsapp_message(
                    sender_number, 
                    f"Send 'start' to begin the {BRAND_NAME} experience!"
                )
                
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}", exc_info=True)
        try:
            send_whatsapp_message(
                sender_number, 
                "Sorry, something went wrong. Please try again by sending 'start'."
            )
            user_states[sender_number] = UserState.IDLE
        except:
            logger.error("Failed to send error message to user")

@app.route('/')
def index():
    return f"{BRAND_NAME} WhatsApp Bot", 200

# FIX: Add proper route for health check
@app.route('/health')
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        
        # Add debugging
        print(f"Verification request received:")
        print(f"Mode: {mode}")
        print(f"Token: {token}")
        print(f"Expected token: {os.getenv('VERIFY_TOKEN')}")
        
        if mode == 'subscribe' and token == os.getenv('VERIFY_TOKEN'):
            return challenge
        else:
            return 'Forbidden', 403
            
    elif request.method == 'POST':
        try:
            data = request.get_json()
            logger.info(f"Received webhook data: {json.dumps(data, indent=2)}")
            
            if 'entry' in data and len(data['entry']) > 0:
                for entry in data['entry']:
                    if 'changes' in entry:
                        for change in entry['changes']:
                            if change.get('field') == 'messages':
                                value = change.get('value', {})
                                messages = value.get('messages', [])
                                
                                for message in messages:
                                    sender_number = message.get('from')
                                    if sender_number:
                                        handle_message(message, sender_number)
            
            return 'OK', 200
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
            return 'Error', 500

# FIX: Update the static file route to properly serve files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename, cache_timeout=0)

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Clean up old files in static directory
    for file in os.listdir('static'):
        if file.startswith('result_'):
            try:
                os.remove(os.path.join('static', file))
            except Exception as e:
                logger.error(f"Error cleaning up file {file}: {str(e)}")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)