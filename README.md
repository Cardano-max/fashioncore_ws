# FashionCore WhatsApp Bot

A WhatsApp bot for virtual try-on using FashionCore technology.

## Deployment

This project is deployed on Railway.app for 24/7 operation.

### Required Files
- `fashioncore.py` - Main application code
- `railway.json` - Railway configuration
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (not committed to git)

### Environment Variables
```
WHATSAPP_API_VERSION=v17.0
PHONE_NUMBER_ID=your_phone_number_id
ACCESS_TOKEN=your_access_token
IMAGE_URL=https://your-railway-app-url.up.railway.app
WEBSITE_URL=https://fashioncore.com/tryon
VERIFY_TOKEN=your_verify_token
```

### Railway Deployment
1. Create a new project on Railway.app
2. Connect your GitHub repository
3. Add environment variables
4. Deploy

## Development
- Python 3.8+
- Flask
- OpenCV
- WhatsApp Business API