# FashionCore Virtual Try-On - 11za Integration Setup Guide

## Overview

This application integrates **11za WhatsApp Business API** with **Kling AI Virtual Try-On** to create a seamless virtual try-on experience via WhatsApp.

## User Flow

1. **Landing Page**: User sees a garment image with a "Try-On" button
2. **WhatsApp Redirect**: Button click opens WhatsApp with pre-selected garment
3. **Person Image Upload**: User is asked to upload their full-body photo
4. **AI Processing**: Kling AI generates virtual try-on (15-20 seconds)
5. **Result Delivery**: User receives the try-on result directly in WhatsApp

## Technical Architecture

### Components

1. **Flask Backend** (`fashioncore_11za.py`)
   - Handles webhooks from 11za
   - Manages conversation state machine
   - Integrates with Kling AI API
   - Serves landing page and admin dashboard

2. **11za WhatsApp API** (`ElevenzaWhatsAppClient`)
   - Sends text messages
   - Sends image messages
   - Receives incoming messages via webhook

3. **Kling AI Virtual Try-On** (`KlingAIClient`)
   - Processes person + garment images
   - Returns realistic try-on results
   - Uses kolors-virtual-try-on-v1-5 model

4. **SQLite Database**
   - Logs all try-on attempts
   - Stores phone numbers, images, and results
   - Exportable to CSV

## Setup Instructions

### 1. Environment Variables

The application automatically creates a `.env` file with these variables:

```env
# 11za WhatsApp API Configuration
ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=<your-11za-auth-token>
ELEVENZA_PHONE_NUMBER=919725791777

# Kling AI Configuration
KLING_ACCESS_KEY=<your-kling-access-key>
KLING_SECRET_KEY=<your-kling-secret-key>

# App Configuration
IMAGE_URL=https://your-app.railway.app
WEBSITE_URL=https://your-website.com
VERIFY_TOKEN=1122
```

**Important**: Update these values in the code or via Railway environment variables:
- `ELEVENZA_AUTH_TOKEN`: Your 11za authentication token
- `ELEVENZA_PHONE_NUMBER`: Your WhatsApp Business number (without +)
- `KLING_ACCESS_KEY` & `KLING_SECRET_KEY`: Your Kling AI credentials
- `IMAGE_URL`: Your Railway deployment URL
- `WEBSITE_URL`: Your frontend website URL

### 2. Configure 11za Webhook

In your 11za dashboard:

1. Go to **Settings** → **Webhooks** or **API Integration**
2. Set webhook URL to: `https://your-app.railway.app/webhook`
3. Set verify token to: `1122` (or your custom token)
4. Enable webhook events for:
   - Incoming messages
   - Image messages
   - Text messages

### 3. Deploy to Railway

```bash
# The railway.json is already configured
# Just push to your Railway-connected git repository

git add .
git commit -m "Add 11za integration"
git push origin claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM
```

### 4. Verify Deployment

After deployment, test these endpoints:

1. **Health Check**: `https://your-app.railway.app/health`
   - Should return: `{"status": "ok", "time": <timestamp>}`

2. **Landing Page**: `https://your-app.railway.app/`
   - Should display garment and Try-On button

3. **Webhook**: Test by sending a WhatsApp message
   - Send "start" to your WhatsApp Business number
   - Bot should respond with welcome message

### 5. Test the Complete Flow

1. Open landing page in browser
2. Click "Try On via WhatsApp" button
3. WhatsApp should open with your business number
4. Send the pre-filled message
5. Bot asks for person photo
6. Upload full-body photo
7. Wait 15-20 seconds
8. Receive try-on result image

## API Integration Details

### 11za WhatsApp API

**Sending Text Message:**
```python
payload = {
    "to": "919999999999",  # Without + sign
    "message": "Your message here"
}
headers = {
    "Content-Type": "application/json",
    "Origin": "https://rangshrii.com/",
    "Authorization": "Bearer <auth-token>"
}
response = requests.post(
    "https://app.11za.in/apis/template/sendTemplate",
    headers=headers,
    json=payload
)
```

**Sending Image Message:**
```python
payload = {
    "to": "919999999999",
    "type": "image",
    "image": {
        "url": "https://example.com/image.jpg",
        "caption": "Your caption"
    }
}
```

**Webhook Format** (expected):
```json
{
  "messages": [
    {
      "from": "919999999999",
      "type": "text",
      "text": "user message"
    }
  ]
}
```

or

```json
{
  "messages": [
    {
      "from": "919999999999",
      "type": "image",
      "image": {
        "url": "https://image-url.com/image.jpg"
      }
    }
  ]
}
```

### Kling AI Virtual Try-On API

**Authentication:**
Uses JWT token with access key and secret key.

**API Call:**
```python
POST https://api.klingai.com/v1/images/kolors-virtual-try-on

{
  "model_name": "kolors-virtual-try-on-v1-5",
  "human_image": "<base64-encoded-person-image>",
  "cloth_image": "<base64-encoded-garment-image>",
  "seed": <random-number>
}
```

**Processing Time:** 15-20 seconds average

**Result Format:**
```json
{
  "data": {
    "task_id": "...",
    "task_status": "succeed",
    "task_result": {
      "images": [
        {
          "url": "https://result-image-url.com/image.jpg"
        }
      ]
    }
  }
}
```

## State Machine

The bot uses a state machine to manage conversations:

```
IDLE → WAITING_FOR_PERSON → PROCESSING → SHOWING_RESULT → IDLE
```

**States:**
- `IDLE`: Initial state, waiting for "start" command
- `WAITING_FOR_PERSON`: Waiting for user to upload person image
- `PROCESSING`: Processing images with Kling AI
- `SHOWING_RESULT`: Result sent, offering to try again

## Database Schema

```sql
CREATE TABLE tryon_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number TEXT NOT NULL,
    person_image_url TEXT,
    garment_image_url TEXT,
    result_image_url TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Admin Dashboard

Access the admin dashboard at: `https://your-app.railway.app/admin`

Features:
- View all try-on attempts
- See phone numbers and timestamps
- View person, garment, and result images
- Download data as CSV

## Customization

### Change Garment Images

Edit `templates/landing.html`:

```html
<img src="YOUR_GARMENT_IMAGE_URL"
     alt="Garment"
     class="garment-image"
     id="garmentImage">
```

### Change Brand Name

Edit in `fashioncore_11za.py`:

```python
BRAND_NAME = "Your Brand Name"
BOT_NAME = "Your Bot Name"
```

### Add Multiple Garments

Create multiple landing pages or add a garment selector:

```python
@app.route('/garment/<garment_id>')
def garment_page(garment_id):
    garment_data = get_garment_from_db(garment_id)
    return render_template('landing.html', garment=garment_data)
```

## Troubleshooting

### 11za Messages Not Sending

1. Verify `ELEVENZA_AUTH_TOKEN` is correct
2. Check `ELEVENZA_PHONE_NUMBER` format (no + sign)
3. Ensure `Origin` header matches your registered domain
4. Check 11za dashboard for API logs

### Webhook Not Receiving Messages

1. Verify webhook URL is accessible: `https://your-app.railway.app/webhook`
2. Check webhook is configured in 11za dashboard
3. Verify `VERIFY_TOKEN` matches in both places
4. Check Railway logs: `railway logs`

### Kling AI Errors

1. Verify `KLING_ACCESS_KEY` and `KLING_SECRET_KEY`
2. Check API quota/limits
3. Ensure images are valid JPG/PNG format
4. Check Kling AI service status

### Images Not Downloading

1. Verify image URLs are publicly accessible
2. Check network connectivity
3. Ensure sufficient disk space
4. Check file permissions

## Monitoring & Logs

### View Logs in Railway

```bash
railway logs --follow
```

### Check Application Logs

Logs are saved to `app.log` with rotation:
- Max size: 10MB per file
- Backup count: 5 files

### Monitor Database

```bash
sqlite3 tryon_data.db "SELECT COUNT(*) FROM tryon_attempts"
```

## Security Considerations

1. **Environment Variables**: Never commit `.env` to git
2. **Auth Tokens**: Rotate tokens regularly
3. **Webhook Verification**: Always verify webhook tokens
4. **Rate Limiting**: Implement rate limiting for production
5. **Image Storage**: Clean up temporary files regularly

## Performance Optimization

1. **Caching**: Add Redis for session state (optional)
2. **Image CDN**: Use CDN for garment images
3. **Background Jobs**: Use Celery for async processing (optional)
4. **Database**: Migrate to PostgreSQL for production scale

## Support & Resources

- **11za Documentation**: Check your 11za dashboard
- **Kling AI Docs**: [Kling AI API Documentation](https://www.klingai.com/)
- **Railway Docs**: [Railway Documentation](https://docs.railway.app/)

## License

Proprietary - All rights reserved

## Contact

For support, contact: support@fashioncore.com
