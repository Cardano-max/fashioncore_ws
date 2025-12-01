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
- **Production Ready**: Deploy to Koyeb free tier (no card required!)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 11za WhatsApp Business account
- Kling AI API credentials (provided)
- Koyeb account (for deployment - **no credit card required!**)

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
2. Set webhook URL: `https://your-app.koyeb.app/webhook` (or your platform URL)
3. Set verify token: `1122`
4. Enable events: Incoming messages, Images, Text

## 🚢 Deployment

### Deploy to Koyeb (Recommended - No Card Required!)

**Quick Start**: See [QUICKSTART_KOYEB.md](./QUICKSTART_KOYEB.md) for 5-minute setup

**Full Guide**: See [KOYEB_DEPLOY.md](./KOYEB_DEPLOY.md) for complete instructions

#### Quick Steps:

1. **Sign up at Koyeb**: https://www.koyeb.com/ (no card required!)
2. **Create App** → Select GitHub → Choose this repository
3. **Configure**: Use Docker builder, set environment variables
4. **Deploy**: Koyeb builds and deploys automatically
5. **Configure Webhook**: Update 11za with your Koyeb URL

```bash
# Or push to auto-deploy
git add .
git commit -m "Deploy virtual try-on bot"
git push origin claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM
```

Koyeb will automatically:
- Build using `Dockerfile`
- Install dependencies
- Deploy with no sleep/wake delays ✅

### Alternative Platforms

- **Render**: See [RENDER_DEPLOY.md](./RENDER_DEPLOY.md) (requires card)
- **Railway**: See [RAILWAY_DEPLOY.md](./RAILWAY_DEPLOY.md) (requires card)

### Post-Deployment

1. **Set environment variables** in Koyeb dashboard (especially `ELEVENZA_AUTH_TOKEN`)
2. **Configure 11za webhook** with your Koyeb URL
3. **Test health check**: `https://your-app.koyeb.app/health`
4. **Test landing page**: `https://your-app.koyeb.app/`
5. **Test WhatsApp flow**: Send "start" to your WhatsApp number

## 📖 Documentation

### Deployment Guides
- **[QUICKSTART_KOYEB.md](./QUICKSTART_KOYEB.md)**: 5-minute Koyeb deployment (no card!)
- **[KOYEB_DEPLOY.md](./KOYEB_DEPLOY.md)**: Complete Koyeb deployment guide
- **[RENDER_DEPLOY.md](./RENDER_DEPLOY.md)**: Render.com deployment guide
- **[RAILWAY_DEPLOY.md](./RAILWAY_DEPLOY.md)**: Railway deployment guide
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

# Test production
python test_webhook.py https://your-app.railway.app
```

### Manual Testing

1. **Health Check**:
   ```bash
   curl https://your-app.koyeb.app/health
   ```

2. **Webhook Verification**:
   ```bash
   curl "https://your-app.koyeb.app/webhook?hub.mode=subscribe&hub.verify_token=1122&hub.challenge=test"
   ```

3. **Landing Page**:
   Open browser to: `https://your-app.koyeb.app/`

4. **Admin Dashboard**:
   Open browser to: `https://your-app.koyeb.app/admin`

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
4. Check deployment platform logs (Koyeb Dashboard → Logs)

### Kling AI Errors

1. Verify API credentials
2. Check API quota
3. Ensure images are valid format
4. Review processing timeout settings

See [INTEGRATION_NOTES.md](./INTEGRATION_NOTES.md) for detailed troubleshooting.

## 📊 Monitoring

### View Logs

```bash
# Koyeb Dashboard → Logs tab
# Real-time logs with filtering

# Local development logs
tail -f app.log
```

### Database Stats

```bash
sqlite3 tryon_data.db "SELECT COUNT(*) as total_attempts FROM tryon_attempts"
```

### Admin Dashboard

Access: `https://your-app.koyeb.app/admin`

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

- **Quick Start**: [QUICKSTART_KOYEB.md](./QUICKSTART_KOYEB.md) - Deploy in 5 minutes!
- **Setup Help**: [SETUP_GUIDE.md](./SETUP_GUIDE.md)
- **API Details**: [INTEGRATION_NOTES.md](./INTEGRATION_NOTES.md)
- **Deployment**: [KOYEB_DEPLOY.md](./KOYEB_DEPLOY.md)
- Check platform logs for deployment issues

## 🎯 Roadmap

- [ ] Multiple garment selection
- [ ] User authentication
- [ ] Save favorite try-ons
- [ ] Share results on social media
- [ ] Video try-on support
- [ ] Multi-language support

---

Built with ❤️ using Flask, 11za, and AI virtual try-on

**Deploy for FREE** on [Koyeb](https://www.koyeb.com/) - No credit card required! 🚀