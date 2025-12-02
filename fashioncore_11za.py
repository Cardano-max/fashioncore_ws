import os
import requests
import cv2
import numpy as np
from flask import Flask, request, send_from_directory, jsonify, render_template, Response
from dotenv import load_dotenv
import base64
import time
import jwt
import logging
import random
import json
import sqlite3
import csv
import io
from datetime import datetime
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

# Create new .env file with correct values for 11za
def create_env_file():
    env_content = """# 11za WhatsApp API Configuration
ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=U2FsdGVkX1/e9ymvz3iAqRt4SA7LgwfStvq6pJdz4WP6yhSMsicFgT7duBMdD9V3q+Qs26KbwdBWtiNeTbqdg8sOO42m2QTejji0oVCKq0Iy81tUHFeqnLqgL285ttgrnk7qY+RRXXaM8taUwCwWVWgIuQxTaoaO4J3/JnxXLoiO8z9TZzNeCuPppwrL+v4A
ELEVENZA_PHONE_NUMBER=917405991551

# AI Service Configuration
KLING_ACCESS_KEY=ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY=pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC

# App Configuration
IMAGE_URL=https://fashioncore-ws-production.up.railway.app
WEBSITE_URL=https://fashioncore-production.up.railway.app
VERIFY_TOKEN=1122

# Test Mode - Set to True to avoid using AI service credits
TEST_MODE=True"""

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
for var in ['ELEVENZA_API_URL', 'ELEVENZA_ORIGIN', 'IMAGE_URL', 'WEBSITE_URL']:
    value = os.getenv(var)
    print(f"{var}: {value}")

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

# Initialize database
DB_PATH = BASE_DIR / 'tryon_data.db'

def init_db():
    """Initialize the SQLite database for storing try-on data."""
    logger.info(f"Initializing database at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS tryon_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_number TEXT NOT NULL,
        person_image_url TEXT,
        garment_image_url TEXT,
        result_image_url TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Call init_db at startup
init_db()

# Constants
MAX_SEED = 999999

# Brand name constants
BRAND_NAME = "FashionCore Magic Try-on"
BOT_NAME = "FashionCore Assistant"

# Test mode flag - if True, will not call actual AI API (saves credits)
TEST_MODE = os.getenv('TEST_MODE', 'True').lower() == 'true'

if TEST_MODE:
    logger.warning("=" * 60)
    logger.warning("🧪 TEST MODE ENABLED - AI API calls will be mocked")
    logger.warning("=" * 60)

# Add these state constants
class UserState:
    IDLE = "idle"
    WAITING_FOR_PERSON = "waiting_for_person"
    PROCESSING = "processing"
    SHOWING_RESULT = "showing_result"

# Add these global dictionaries
user_states = {}  # Format: {phone_number: UserState}
user_images = {}  # Format: {phone_number: {'person': image_path, 'garment': garment_url}}
user_results = {}  # Format: {phone_number: {'result_url': url}}
user_last_activity = {}  # Format: {phone_number: timestamp}

# Store garment selections from landing page
garment_selections = {}  # Format: {session_id: garment_image_url}

# Trigger words that bot responds to
TRIGGER_WORDS = ['start', 'hi', 'hello', 'hey', 'begin', 'tryon', 'try on', 'help']

# Session timeout in seconds (10 minutes)
SESSION_TIMEOUT = 600

class AITryOnClient:
    """Client for AI-powered virtual try-on service"""
    def __init__(self):
        self.access_key = os.getenv('KLING_ACCESS_KEY', 'ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP')
        self.secret_key = os.getenv('KLING_SECRET_KEY', 'pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC')
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

    def try_on(self, person_img: np.ndarray, garment_img: np.ndarray, seed: int) -> Tuple[np.ndarray, str, str]:
        """
        Use the AI Virtual Try-on service to generate a try-on image.

        Args:
            person_img: The person's image
            garment_img: The garment image
            seed: Random seed for generation

        Returns:
            The resulting image, original URL, and status message
        """
        if person_img is None or garment_img is None:
            raise ValueError("Empty image")

        # TEST MODE: Return mock result without calling API
        if TEST_MODE:
            self.logger.info("🧪 TEST MODE: Mocking AI response (no API call)")
            self.logger.info("⏱️  Simulating processing time...")
            time.sleep(3)  # Simulate processing time

            # Use a sample result image URL
            mock_result_url = "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=800&h=1000&fit=crop"

            self.logger.info(f"✅ Mock try-on successful! Result URL: {mock_result_url}")

            # Create a simple mock result image (blend person and garment)
            result_img = cv2.addWeighted(person_img, 0.6, garment_img, 0.4, 0)

            return result_img, mock_result_url, "Success (TEST MODE)"

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
            self.logger.info("Making API request to Virtual Try-on service")
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=data,
                timeout=50
            )

            if response.status_code == 429:
                error_msg = "Sorry, our service is currently at capacity. Please try again in a few minutes."
                self.logger.error(f"API rate limit exceeded: {response.text}")
                return None, None, error_msg

            if response.status_code != 200:
                error_msg = f"Error: API returned status code {response.status_code}"
                self.logger.error(f"API error: {response.text}")
                return None, None, error_msg

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
                        return result_img, output_url, "Success"
                    elif status == "failed":
                        error_msg = f"Sorry, we couldn't create the try-on image. {result['data']['task_status_msg']}"
                        self.logger.error(f"Task failed: {result['data']['task_status_msg']}")
                        return None, None, error_msg
                    else:
                        self.logger.info(f"Task status: {status}. Waiting...")

                except requests.exceptions.ReadTimeout:
                    self.logger.warning(f"Timeout on attempt {attempt+1}/12. Retrying...")
                    if attempt == 11:
                        return None, None, "Sorry, the try-on is taking longer than expected. Please try again."

                time.sleep(1)

            return None, None, "The try-on is taking too long. Please try again later."

        except requests.exceptions.RequestException as e:
            error_msg = f"Sorry, we're having trouble connecting to our service. Please try again later."
            self.logger.error(f"API error: {str(e)}")
            return None, None, error_msg
        except Exception as e:
            error_msg = f"Sorry, something went wrong. Please try again later."
            self.logger.error(f"Unexpected error: {str(e)}")
            return None, None, error_msg

class ElevenzaWhatsAppClient:
    """Client for 11za (Elevenza) WhatsApp API"""

    def __init__(self):
        self.api_url = os.getenv('ELEVENZA_API_URL')
        self.origin = os.getenv('ELEVENZA_ORIGIN')
        self.auth_token = os.getenv('ELEVENZA_AUTH_TOKEN')
        self.phone_number = os.getenv('ELEVENZA_PHONE_NUMBER')
        self.logger = logging.getLogger(__name__)

    def _get_headers(self) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Origin': self.origin,
            'authToken': self.auth_token
        }

    def send_text_message(self, to_number: str, message: str) -> bool:
        """Send a text message via 11za API"""
        try:
            # Format phone number if needed (remove + and country code if present)
            formatted_number = to_number.replace('+', '')

            payload = {
                "sendto": formatted_number,
                "authToken": self.auth_token,
                "originWebsite": self.origin,
                "message": message
            }

            self.logger.info(f"Sending text message to {formatted_number}")
            self.logger.debug(f"Payload: {json.dumps(payload)}")

            response = requests.post(
                self.api_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )

            self.logger.info(f"Response status: {response.status_code}")
            self.logger.debug(f"Response: {response.text}")

            if response.status_code == 200:
                self.logger.info(f"Message sent successfully to {formatted_number}")
                return True
            else:
                self.logger.error(f"Failed to send message: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending text message: {str(e)}", exc_info=True)
            return False

    def send_image_message(self, to_number: str, image_url: str, caption: str = "") -> bool:
        """Send an image message via 11za API"""
        try:
            formatted_number = to_number.replace('+', '')

            payload = {
                "sendto": formatted_number,
                "authToken": self.auth_token,
                "originWebsite": self.origin,
                "type": "image",
                "myfile": image_url,  # 11za uses 'myfile' for image URL
                "caption": caption
            }

            self.logger.info(f"Sending image to {formatted_number}")
            self.logger.debug(f"Image URL: {image_url}")

            response = requests.post(
                self.api_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )

            self.logger.info(f"Response status: {response.status_code}")

            if response.status_code == 200:
                self.logger.info(f"Image sent successfully to {formatted_number}")
                return True
            else:
                self.logger.error(f"Failed to send image: {response.text}")
                # Fallback to text message with link
                fallback_msg = f"{caption}\n\nView your image: {image_url}"
                return self.send_text_message(to_number, fallback_msg)

        except Exception as e:
            self.logger.error(f"Error sending image: {str(e)}", exc_info=True)
            # Fallback to text message
            try:
                fallback_msg = f"{caption}\n\nView your image: {image_url}"
                return self.send_text_message(to_number, fallback_msg)
            except:
                return False

# Initialize 11za client
elevenza_client = ElevenzaWhatsAppClient()

def log_tryon_attempt(phone_number, person_image_url, garment_image_url, result_image_url):
    """Log a try-on attempt to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('''
        INSERT INTO tryon_attempts
        (phone_number, person_image_url, garment_image_url, result_image_url)
        VALUES (?, ?, ?, ?)
        ''', (phone_number, person_image_url, garment_image_url, result_image_url))

        conn.commit()
        conn.close()
        logger.info(f"Logged try-on attempt for {phone_number} to database")
        return True
    except Exception as e:
        logger.error(f"Error logging try-on attempt to database: {str(e)}")
        return False

def download_image_from_url(image_url: str) -> Optional[str]:
    """Download image from URL and save locally"""
    logger.info(f"Starting download for image URL: {image_url}")

    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            logger.error(f"Failed to download image: {response.status_code}")
            return None

        filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
        with open(filename, 'wb') as f:
            f.write(response.content)

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
        logger.info("Initializing AI processing client")
        client = AITryOnClient()

        # Process images
        logger.info("Calling Virtual Try-on service")
        result_img, direct_url, status_message = client.try_on(person_img, garment_img, random.randint(0, MAX_SEED))

        if result_img is None:
            logger.error(f"Processing failed. Status: {status_message}")
            return None, status_message

        # Return the direct URL from AI service
        if direct_url:
            logger.info(f"Using direct URL from AI service: {direct_url}")
            return direct_url, "Success"

        logger.error("No direct URL available from AI service")
        return None, "Sorry, we couldn't generate a shareable image URL. Please try again."

    except Exception as e:
        logger.error(f"Error processing images: {str(e)}", exc_info=True)
        return None, "Sorry, something went wrong while generating your try-on. Please try again later."

def check_session_timeout(sender_number: str) -> bool:
    """Check if user session has timed out"""
    if sender_number in user_last_activity:
        elapsed = time.time() - user_last_activity[sender_number]
        if elapsed > SESSION_TIMEOUT:
            logger.info(f"Session timeout for {sender_number}")
            reset_user_session(sender_number)
            return True
    return False

def reset_user_session(sender_number: str):
    """Reset user session completely"""
    logger.info(f"Resetting session for {sender_number}")
    user_states.pop(sender_number, None)
    user_images.pop(sender_number, None)
    user_results.pop(sender_number, None)
    user_last_activity.pop(sender_number, None)

def update_user_activity(sender_number: str):
    """Update last activity timestamp for user"""
    user_last_activity[sender_number] = time.time()

def is_trigger_word(text: str) -> bool:
    """Check if text contains any trigger word"""
    text_lower = text.lower().strip()
    # Check for exact match or if trigger word is in the text
    for trigger in TRIGGER_WORDS:
        if trigger in text_lower:
            return True
    # Also check for start_ pattern
    if text_lower.startswith('start_'):
        return True
    return False

def handle_message(message: dict, sender_number: str):
    """Handle incoming messages from 11za webhook"""
    try:
        # Check for session timeout
        check_session_timeout(sender_number)

        # Update last activity
        update_user_activity(sender_number)

        current_state = user_states.get(sender_number, UserState.IDLE)
        message_type = message.get('type', 'text')
        logger.info(f"Handling message from {sender_number}. Type: {message_type}, State: {current_state}")

        if message_type == 'text':
            text = message.get('text', '').lower().strip()

            # Ignore empty messages
            if not text:
                logger.info(f"Ignoring empty message from {sender_number}")
                return

            # Check if this is a start command with garment ID
            if text.startswith('start_'):
                garment_id = text.replace('start_', '')
                logger.info(f"User starting with garment ID: {garment_id}")

                # Look up garment URL from selection
                garment_url = garment_selections.get(garment_id)
                if garment_url:
                    user_images[sender_number] = {'garment_url': garment_url}
                    logger.info(f"Found garment URL: {garment_url}")

                user_states[sender_number] = UserState.WAITING_FOR_PERSON
                update_user_activity(sender_number)
                elevenza_client.send_text_message(
                    sender_number,
                    f"👋 Welcome to {BRAND_NAME}! Let's create a stunning virtual outfit for you.\n\nPlease send a full-body photo of yourself standing straight against a plain background."
                )
                return

            # In IDLE state, only respond to trigger words
            if current_state == UserState.IDLE:
                if is_trigger_word(text):
                    user_states[sender_number] = UserState.WAITING_FOR_PERSON
                    update_user_activity(sender_number)
                    elevenza_client.send_text_message(
                        sender_number,
                        f"👋 Welcome to {BRAND_NAME}! Send a full-body photo to begin."
                    )
                else:
                    # Ignore non-trigger messages in IDLE state
                    logger.info(f"Ignoring non-trigger message in IDLE state from {sender_number}: {text}")
                return

            # In WAITING_FOR_PERSON state, remind to send image
            elif current_state == UserState.WAITING_FOR_PERSON:
                elevenza_client.send_text_message(
                    sender_number,
                    "Please send a full-body photo of yourself to continue. 📸"
                )
                return

            # In PROCESSING state, tell user to wait
            elif current_state == UserState.PROCESSING:
                elevenza_client.send_text_message(
                    sender_number,
                    "Please wait, we're creating your try-on... ⏳"
                )
                return

            # In SHOWING_RESULT state, check if user wants to start again
            elif current_state == UserState.SHOWING_RESULT:
                if is_trigger_word(text):
                    # Reset and start new session
                    reset_user_session(sender_number)
                    user_states[sender_number] = UserState.WAITING_FOR_PERSON
                    update_user_activity(sender_number)
                    elevenza_client.send_text_message(
                        sender_number,
                        f"👋 Great! Send a full-body photo to try another outfit."
                    )
                else:
                    # After showing result, auto-reset and go to IDLE
                    logger.info(f"Auto-resetting session after result shown for {sender_number}")
                    reset_user_session(sender_number)
                return

        elif message_type == 'image':
            if current_state == UserState.WAITING_FOR_PERSON:
                # Handle person image
                image_url = message.get('image', {}).get('url')
                if not image_url:
                    elevenza_client.send_text_message(
                        sender_number,
                        "I couldn't process that image. Please try sending it again."
                    )
                    return

                image_path = download_image_from_url(image_url)
                if image_path:
                    # Check if we have a pre-selected garment
                    if sender_number in user_images and 'garment_url' in user_images[sender_number]:
                        garment_url = user_images[sender_number]['garment_url']
                        logger.info(f"Using pre-selected garment: {garment_url}")

                        # Download garment image
                        garment_path = download_image_from_url(garment_url)
                        if not garment_path:
                            elevenza_client.send_text_message(
                                sender_number,
                                "Sorry, there was an issue with the garment image. Please try again."
                            )
                            return

                        # Let the user know we're processing
                        elevenza_client.send_text_message(
                            sender_number,
                            f"✨ Creating your outfit with {BRAND_NAME} magic! This should take about 15-20 seconds..."
                        )

                        user_states[sender_number] = UserState.PROCESSING

                        try:
                            # Process the images
                            direct_url, status_message = process_images(image_path, garment_path)

                            if direct_url:
                                # Save the result URL
                                user_results[sender_number] = {'result_url': direct_url}

                                # Log this try-on attempt
                                log_tryon_attempt(
                                    sender_number,
                                    image_url,
                                    garment_url,
                                    direct_url
                                )

                                # Send the result image
                                success = elevenza_client.send_image_message(
                                    sender_number,
                                    direct_url,
                                    f"✨ Here's your {BRAND_NAME} result! What do you think?"
                                )

                                if success:
                                    user_states[sender_number] = UserState.SHOWING_RESULT
                                    time.sleep(2)
                                    elevenza_client.send_text_message(
                                        sender_number,
                                        "Love it? Want to try another outfit? Send 'start' to try again! 😊"
                                    )
                                    # Auto-reset session after 5 seconds
                                    time.sleep(5)
                                    logger.info(f"Auto-resetting session for {sender_number} after result")
                                    reset_user_session(sender_number)
                                else:
                                    elevenza_client.send_text_message(
                                        sender_number,
                                        f"I created your try-on image! View it at: {direct_url}\n\nWant to try more? Send 'start'!"
                                    )
                                    # Auto-reset session
                                    reset_user_session(sender_number)
                            else:
                                elevenza_client.send_text_message(
                                    sender_number,
                                    f"Sorry, {status_message} Please try again by sending 'start'."
                                )
                                reset_user_session(sender_number)
                        except Exception as e:
                            logger.error(f"Error in processing: {str(e)}", exc_info=True)
                            elevenza_client.send_text_message(
                                sender_number,
                                "Sorry, something went wrong while creating your try-on. Please try again by sending 'start'."
                            )
                            reset_user_session(sender_number)
                        finally:
                            # Clean up temporary files
                            try:
                                os.remove(image_path)
                                os.remove(garment_path)
                            except Exception as e:
                                logger.error(f"Error removing temporary files: {str(e)}")
                    else:
                        # No garment selected, need to ask for one
                        user_images[sender_number] = {'person': image_path, 'person_url': image_url}
                        elevenza_client.send_text_message(
                            sender_number,
                            "Great! Now please send an image of the clothing item you want to try on."
                        )
                else:
                    elevenza_client.send_text_message(
                        sender_number,
                        "I'm having trouble downloading your image. Please try sending a different photo."
                    )
            else:
                # Image sent in wrong state, ignore
                logger.info(f"Ignoring image in state {current_state} from {sender_number}")
                if current_state == UserState.IDLE:
                    elevenza_client.send_text_message(
                        sender_number,
                        f"Send 'start' to begin the {BRAND_NAME} experience!"
                    )

        else:
            # Ignore other message types (status, reactions, etc.)
            logger.info(f"Ignoring message type '{message_type}' from {sender_number}")

    except Exception as e:
        logger.error(f"Error handling message: {str(e)}", exc_info=True)
        try:
            elevenza_client.send_text_message(
                sender_number,
                "Sorry, something went wrong. Please try again by sending 'start'."
            )
            reset_user_session(sender_number)
        except:
            logger.error("Failed to send error message to user")

@app.route('/')
def index():
    """Landing page with garment selection"""
    return render_template('landing.html')

@app.route('/select-garment/<garment_id>')
def select_garment(garment_id):
    """Handle garment selection from landing page"""
    # Get garment URL from query params
    garment_url = request.args.get('url')
    if garment_url:
        garment_selections[garment_id] = garment_url
        logger.info(f"Stored garment selection: {garment_id} -> {garment_url}")

    # Generate WhatsApp link
    phone_number = os.getenv('ELEVENZA_PHONE_NUMBER', '917405991551')
    message = f"start_{garment_id}"
    whatsapp_url = f"https://wa.me/{phone_number}?text={message}"

    return jsonify({
        'success': True,
        'whatsapp_url': whatsapp_url
    })

# Add health check route
@app.route('/health')
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200

@app.route('/admin')
def admin_panel():
    """Simple admin panel to view try-on data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute('SELECT * FROM tryon_attempts ORDER BY timestamp DESC')
        attempts = c.fetchall()

        attempts_data = []
        for attempt in attempts:
            attempt_dict = dict(attempt)
            if attempt_dict['timestamp']:
                try:
                    timestamp = datetime.strptime(attempt_dict['timestamp'], '%Y-%m-%d %H:%M:%S')
                    attempt_dict['formatted_time'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    attempt_dict['formatted_time'] = attempt_dict['timestamp']
            else:
                attempt_dict['formatted_time'] = 'Unknown'

            attempts_data.append(attempt_dict)

        conn.close()

        return render_template('admin.html', attempts=attempts_data)
    except Exception as e:
        logger.error(f"Error rendering admin page: {str(e)}")
        return f"Error rendering admin page: {str(e)}", 500

@app.route('/download-csv')
def download_csv():
    """Generate and download CSV of all try-on data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute('SELECT * FROM tryon_attempts ORDER BY timestamp DESC')
        attempts = c.fetchall()

        conn.close()

        csv_data = io.StringIO()
        csv_writer = csv.writer(csv_data)

        csv_writer.writerow(['ID', 'Phone Number', 'Person Image URL', 'Garment Image URL',
                            'Result Image URL', 'Timestamp'])

        for attempt in attempts:
            csv_writer.writerow([
                attempt['id'],
                attempt['phone_number'],
                attempt['person_image_url'],
                attempt['garment_image_url'],
                attempt['result_image_url'],
                attempt['timestamp']
            ])

        output = csv_data.getvalue()
        csv_data.close()

        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition":
                     f"attachment; filename=tryon_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}")
        return f"Error generating CSV: {str(e)}", 500

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """Webhook endpoint for 11za"""
    if request.method == 'GET':
        # Verification endpoint
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        logger.info(f"Webhook verification request - Mode: {mode}, Token: {token}")

        if mode == 'subscribe' and token == os.getenv('VERIFY_TOKEN'):
            logger.info("Webhook verified successfully")
            return challenge
        else:
            logger.error("Webhook verification failed")
            return 'Forbidden', 403

    elif request.method == 'POST':
        try:
            data = request.get_json()
            logger.info(f"Received webhook data: {json.dumps(data, indent=2)}")

            # Handle 11za direct webhook format
            if 'from' in data and 'content' in data:
                sender_number = data.get('from')
                content = data.get('content', {})
                content_type = content.get('contentType', 'text')

                # Transform 11za format to internal format
                message = {}
                if content_type == 'text':
                    message['type'] = 'text'
                    message['text'] = content.get('text', '')
                elif content_type == 'image':
                    message['type'] = 'image'
                    message['image'] = {
                        'url': content.get('url', '')
                    }

                if sender_number and message:
                    logger.info(f"Processing 11za message from {sender_number}: {message}")
                    handle_message(message, sender_number)

            # Handle Meta-style message array format (if 11za uses it)
            elif 'messages' in data:
                messages = data['messages']
                for message in messages:
                    sender_number = message.get('from')
                    if sender_number:
                        handle_message(message, sender_number)

            # Handle Meta-style entry format (if 11za uses it)
            elif 'entry' in data:
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

# Create template files
def create_template_files():
    """Create necessary template files"""
    os.makedirs('templates', exist_ok=True)

    # Create landing page
    landing_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FashionCore - Virtual Try-On</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            max-width: 500px;
            width: 100%;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            text-align: center;
            color: white;
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .content {
            padding: 30px;
        }

        .garment-display {
            text-align: center;
            margin-bottom: 30px;
        }

        .garment-image {
            width: 100%;
            max-width: 350px;
            height: 350px;
            object-fit: cover;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }

        .garment-name {
            font-size: 22px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }

        .garment-description {
            font-size: 14px;
            color: #666;
            line-height: 1.6;
        }

        .try-on-button {
            display: block;
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            text-decoration: none;
            border-radius: 50px;
            font-size: 18px;
            font-weight: bold;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }

        .try-on-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        }

        .try-on-button:active {
            transform: translateY(0);
        }

        .features {
            margin-top: 30px;
            padding-top: 30px;
            border-top: 1px solid #eee;
        }

        .feature {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            color: #666;
            font-size: 14px;
        }

        .feature-icon {
            width: 24px;
            height: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            margin-right: 12px;
            flex-shrink: 0;
        }

        @media (max-width: 600px) {
            .header h1 {
                font-size: 24px;
            }

            .garment-image {
                height: 280px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✨ FashionCore Magic Try-On</h1>
            <p>See how this outfit looks on you instantly!</p>
        </div>

        <div class="content">
            <div class="garment-display">
                <img src="https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=500&h=500&fit=crop"
                     alt="Stylish Garment"
                     class="garment-image"
                     id="garmentImage">
                <div class="garment-name">Elegant Summer Dress</div>
                <div class="garment-description">
                    Experience this beautiful outfit virtually before you buy.
                    Our AI-powered try-on technology creates realistic previews in seconds!
                </div>
            </div>

            <a href="#" class="try-on-button" id="tryOnBtn">
                🪄 Try On via WhatsApp
            </a>

            <div class="features">
                <div class="feature">
                    <div class="feature-icon">✓</div>
                    <div>Instant AI-powered virtual try-on</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✓</div>
                    <div>Realistic and accurate results</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✓</div>
                    <div>Simple WhatsApp integration</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('tryOnBtn').addEventListener('click', function(e) {
            e.preventDefault();

            // Generate unique session ID
            const sessionId = 'garment_' + Date.now();

            // Get garment image URL
            const garmentUrl = document.getElementById('garmentImage').src;

            // Call backend to store garment selection and get WhatsApp link
            fetch('/select-garment/' + sessionId + '?url=' + encodeURIComponent(garmentUrl))
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.href = data.whatsapp_url;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Something went wrong. Please try again.');
                });
        });
    </script>
</body>
</html>"""

    with open('templates/landing.html', 'w') as f:
        f.write(landing_template)

    # Create admin template (reuse existing one with minor updates)
    admin_template = """<!DOCTYPE html>
<html>
<head>
    <title>FashionCore Try-on Admin</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .button {
            display: inline-block;
            background-color: #667eea;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #667eea;
            color: white;
            position: sticky;
            top: 0;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .image-link {
            color: #667eea;
            text-decoration: none;
        }
        .image-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 FashionCore Try-on Admin Panel</h1>
        <a href="/download-csv" class="button">📥 Download All Data (CSV)</a>

        <h2>Try-on Attempts</h2>

        {% if attempts %}
            <table>
                <tr>
                    <th>ID</th>
                    <th>Phone Number</th>
                    <th>Time</th>
                    <th>Person Image</th>
                    <th>Garment Image</th>
                    <th>Result Image</th>
                </tr>
                {% for attempt in attempts %}
                    <tr>
                        <td>{{ attempt.id }}</td>
                        <td>{{ attempt.phone_number }}</td>
                        <td>{{ attempt.formatted_time }}</td>
                        <td>
                            {% if attempt.person_image_url %}
                                <a href="{{ attempt.person_image_url }}" target="_blank" class="image-link">View</a>
                            {% else %}
                                N/A
                            {% endif %}
                        </td>
                        <td>
                            {% if attempt.garment_image_url %}
                                <a href="{{ attempt.garment_image_url }}" target="_blank" class="image-link">View</a>
                            {% else %}
                                N/A
                            {% endif %}
                        </td>
                        <td>
                            {% if attempt.result_image_url %}
                                <a href="{{ attempt.result_image_url }}" target="_blank" class="image-link">View Result</a>
                            {% else %}
                                N/A
                            {% endif %}
                        </td>
                    </tr>
                {% endfor %}
            </table>
        {% else %}
            <p>No try-on attempts recorded yet.</p>
        {% endif %}
    </div>
</body>
</html>"""

    with open('templates/admin.html', 'w') as f:
        f.write(admin_template)

    logger.info("Created template files")

# Create template files on startup
create_template_files()

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
