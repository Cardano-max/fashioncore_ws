# FashionCore Virtual Try-On - 11za Integration

🎨 An AI-powered virtual try-on experience via WhatsApp using **11za API** and **Kling AI**.

## 🌟 Features

- **Landing Page**: Beautiful product showcase with "Try-On" button
- **WhatsApp Integration**: Seamless 11za WhatsApp Business API integration
- **AI Virtual Try-On**: Powered by AI service (fully white-labeled)
- **Smart Conversation Flow**: State machine-based chat management with trigger words
- **Edge Case Handling**: Auto-reset, session timeout, message filtering
- **Admin Dashboard**: Track all try-on attempts and user interactions
- **Database Logging**: SQLite database with CSV export capability
- **TEST_MODE**: Test full flow without AI charges
- **Production Ready**: Deploy with ngrok or cloud platforms

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 11za WhatsApp Business account
- Kling AI API credentials (provided)
- ngrok account (for testing - **100% free, no credit card!**) OR cloud platform with card

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd fashioncore_ws
   pip install -r requirements.txt
   ```

2. **Configure environment** (auto-created, but verify):
   ```bash
   # Edit fashioncore_11za.py and update:
   # - ELEVENZA_AUTH_TOKEN
   # - ELEVENZA_PHONE_NUMBER
   # - KLING_ACCESS_KEY
   # - KLING_SECRET_KEY
   ```

3. **Run locally**:
   ```bash
   python fashioncore_11za.py
   ```

4. **Test**:
   ```bash
   python test_webhook.py
   ```

5. **Open browser**: http://localhost:8080

## 📋 User Flow

```
┌─────────────────┐
│  Landing Page   │ User sees garment + "Try-On" button
└────────┬────────┘
         │ Click button
         ▼
┌─────────────────┐
│    WhatsApp     │ Opens with pre-selected garment
└────────┬────────┘
         │ Bot: "Send your photo"
         ▼
┌─────────────────┐
│  Upload Photo   │ User sends full-body image
└────────┬────────┘
         │ Validate image
         ▼
┌─────────────────┐
│  AI Processing  │ Kling AI generates try-on (15-20s)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Send Result    │ User receives try-on image
└─────────────────┘
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Landing Page                       │
│              (templates/landing.html)                │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│              Flask Backend                           │
│         (fashioncore_11za.py)                       │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────┐         │
│  │ 11za WhatsApp   │  │  Kling AI        │         │
│  │ Client          │  │  Client          │         │
│  └─────────────────┘  └──────────────────┘         │
│                                                      │
│  ┌──────────────────────────────────────┐          │
│  │     State Machine Manager             │          │
│  │  (User States & Conversation Flow)    │          │
│  └──────────────────────────────────────┘          │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│            SQLite Database                           │
│         (tryon_data.db)                             │
└──────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables

Set these in your deployment platform (Koyeb, Render, etc.):

```env
# 11za WhatsApp API
ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=<your-token>
ELEVENZA_PHONE_NUMBER=919725791777

# AI Service Credentials
KLING_ACCESS_KEY=<your-key>
KLING_SECRET_KEY=<your-secret>

# Application
VERIFY_TOKEN=1122
TEST_MODE=True

# Port (platform will set automatically)
PORT=8080
```

### 11za Webhook Setup

1. Go to 11za Dashboard → Settings → Webhooks
2. Set webhook URL: `https://your-url.ngrok.io/webhook` (or your platform URL)
3. Set verify token: `1122`
4. Enable events: Incoming messages, Images, Text

## 🚢 Deployment

### 🎯 Best Option: ngrok (100% Free, No Card!)

**⭐ RECOMMENDED for testing without credit card**

**Quick Start**: See [QUICKSTART_NGROK.md](./QUICKSTART_NGROK.md) for 5-minute setup

#### Why ngrok?
- ✅ **100% free** - no credit card ever required
- ✅ **No restrictions** - unlimited API calls
- ✅ **Easy debugging** - see logs in real-time
- ✅ **Instant setup** - running in 5 minutes
- ⚠️ **Trade-off**: Your computer must stay on

#### Quick Steps:

```bash
# 1. Install ngrok
brew install ngrok  # Mac
# or download from https://ngrok.com/

# 2. Sign up (free, no card!)
# Get authtoken from https://dashboard.ngrok.com/

# 3. Connect authtoken
ngrok config add-authtoken YOUR_AUTH_TOKEN

# 4. Run your bot
python fashioncore_11za.py

# 5. In new terminal, expose to web
ngrok http 8080

# 6. Copy the HTTPS URL and configure 11za webhook!
```

**Full Guide**: [QUICKSTART_NGROK.md](./QUICKSTART_NGROK.md)

---

### Cloud Platforms (Require Credit Card)

⚠️ **Reality Check**: As of 2025, nearly all "free" cloud platforms require credit card verification to prevent abuse.

**Glitch shut down** project hosting in July 2025.

#### Available Options:

**For Production (24/7 uptime):**

- **Koyeb**: See [KOYEB_DEPLOY.md](./KOYEB_DEPLOY.md) ⚠️ *Card required*
  - Best performance (no sleep)
  - Docker support
  - Free tier with card verification

- **Render**: See [RENDER_DEPLOY.md](./RENDER_DEPLOY.md) ⚠️ *Card required*
  - 750 hours/month free
  - Sleeps after 15 min
  - Card verification needed

- **Railway**: See [RAILWAY_DEPLOY.md](./RAILWAY_DEPLOY.md) ⚠️ *Card required*
  - Limited free tier
  - Card verification needed

**Note**: If you don't have a credit card, use **ngrok** for testing. When ready for production, you'll need card-verified hosting or paid plan ($5-10/month).

---

### Post-Deployment (ngrok)

1. **Run bot locally**: `python fashioncore_11za.py`
2. **Start ngrok**: `ngrok http 8080`
3. **Copy HTTPS URL** from ngrok output
4. **Configure 11za webhook** with ngrok URL
5. **Test health check**: `https://your-url.ngrok.io/health`
6. **Test WhatsApp flow**: Send "start" to your WhatsApp number

### Post-Deployment (Cloud)

1. **Set environment variables** in platform dashboard (especially `ELEVENZA_AUTH_TOKEN`)
2. **Configure 11za webhook** with your cloud URL
3. **Test health check**: `https://your-app.platform.com/health`
4. **Test landing page**: `https://your-app.platform.com/`
5. **Test WhatsApp flow**: Send "start" to your WhatsApp number

## 📖 Documentation

### Deployment Guides
- **[QUICKSTART_NGROK.md](./QUICKSTART_NGROK.md)**: ⭐ 5-minute ngrok setup (no card!)
- **[PYTHONANYWHERE_DEPLOY.md](./PYTHONANYWHERE_DEPLOY.md)**: PythonAnywhere info (API restrictions)
- **[KOYEB_DEPLOY.md](./KOYEB_DEPLOY.md)**: Koyeb deployment (requires card)
- **[RENDER_DEPLOY.md](./RENDER_DEPLOY.md)**: Render deployment (requires card)
- **[RAILWAY_DEPLOY.md](./RAILWAY_DEPLOY.md)**: Railway deployment (requires card)
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)**: Complete feature documentation

### Setup & Testing
- **[SETUP_GUIDE.md](./SETUP_GUIDE.md)**: Complete setup and configuration guide
- **[INTEGRATION_NOTES.md](./INTEGRATION_NOTES.md)**: 11za API integration notes and troubleshooting
- **[test_webhook.py](./test_webhook.py)**: Test suite for all endpoints

## 🧪 Testing

### Run Test Suite

```bash
# Test local instance
python test_webhook.py

# Test production (use your ngrok or platform URL)
python test_webhook.py https://your-url.ngrok.io
```

### Manual Testing

1. **Health Check**:
   ```bash
   curl https://your-url.ngrok.io/health
   ```

2. **Webhook Verification**:
   ```bash
   curl "https://your-url.ngrok.io/webhook?hub.mode=subscribe&hub.verify_token=1122&hub.challenge=test"
   ```

3. **Landing Page**:
   Open browser to: `https://your-url.ngrok.io/`

4. **Admin Dashboard**:
   Open browser to: `https://your-url.ngrok.io/admin`

## 🎨 Customization

### Change Garment Images

Edit `templates/landing.html`:
```html
<img src="YOUR_GARMENT_URL" class="garment-image" id="garmentImage">
```

### Change Brand Name

Edit `fashioncore_11za.py`:
```python
BRAND_NAME = "Your Brand Name"
BOT_NAME = "Your Bot Name"
```

### Add Multiple Garments

Create a product catalog and route:
```python
@app.route('/product/<product_id>')
def product_page(product_id):
    product = get_product(product_id)
    return render_template('landing.html', product=product)
```

## 🐛 Troubleshooting

### Messages Not Sending

1. Check 11za auth token
2. Verify phone number format (no + sign)
3. Check 11za dashboard logs
4. Review `app.log` for errors

### Webhook Not Receiving

1. Verify webhook URL is publicly accessible
2. Check verify token matches
3. Test with `test_webhook.py`
4. Check deployment platform logs (ngrok: terminal output, Cloud: dashboard logs)

### Kling AI Errors

1. Verify API credentials
2. Check API quota
3. Ensure images are valid format
4. Review processing timeout settings

See [INTEGRATION_NOTES.md](./INTEGRATION_NOTES.md) for detailed troubleshooting.

## 📊 Monitoring

### View Logs

```bash
# ngrok: Logs appear in terminal where bot is running
# ngrok web interface: http://127.0.0.1:4040

# Cloud platforms: Dashboard → Logs tab

# Local development logs
tail -f app.log
```

### Database Stats

```bash
sqlite3 tryon_data.db "SELECT COUNT(*) as total_attempts FROM tryon_attempts"
```

### Admin Dashboard

Access: `https://your-url.ngrok.io/admin` (or your platform URL)

Features:
- View all try-on attempts
- Export data to CSV
- Monitor user engagement

## 🔐 Security

- ✅ Environment variables for sensitive data
- ✅ Webhook token verification
- ✅ Origin header validation
- ✅ Image URL validation
- ✅ Automatic cleanup of temporary files

## 📈 Performance

- **Average processing time**: 15-20 seconds
- **Kling AI model**: kolors-virtual-try-on-v1-5
- **Database**: SQLite (upgrade to PostgreSQL for scale)
- **Cleanup**: Automatic temporary file removal

## 🤝 Contributing

This is a proprietary project. For contributions:

1. Create a feature branch
2. Make changes with tests
3. Submit pull request
4. Wait for review

## 📄 License

Proprietary - All rights reserved

## 💬 Support

- **Quick Start**: [QUICKSTART_NGROK.md](./QUICKSTART_NGROK.md) - Deploy in 5 minutes (no card!)
- **Setup Help**: [SETUP_GUIDE.md](./SETUP_GUIDE.md)
- **API Details**: [INTEGRATION_NOTES.md](./INTEGRATION_NOTES.md)
- **Cloud Deployment**: See Koyeb/Render/Railway guides (require card)
- **Platform Alternatives**: [PYTHONANYWHERE_DEPLOY.md](./PYTHONANYWHERE_DEPLOY.md)
- Check logs: ngrok terminal or cloud dashboard

## 🎯 Roadmap

- [ ] Multiple garment selection
- [ ] User authentication
- [ ] Save favorite try-ons
- [ ] Share results on social media
- [ ] Video try-on support
- [ ] Multi-language support

---

Built with ❤️ using Flask, 11za, and AI virtual try-on

**Deploy for FREE** with [ngrok](https://ngrok.com/) - No credit card required! 🚀