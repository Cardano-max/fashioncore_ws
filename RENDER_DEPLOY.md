# 🚀 Deploy to Render.com (Free Tier)

## Why Render?

✅ **Completely Free** - 750 hours/month (24/7 uptime)
✅ **Always On** - No sleep/wake delays
✅ **Auto-Deploy** - Connect GitHub, auto-deploy on push
✅ **HTTPS** - Automatic SSL certificates
✅ **Easy Setup** - 5 minute deployment

---

## 📋 Step-by-Step Deployment

### Step 1: Create Render Account

1. Go to https://render.com/
2. Click **"Get Started for Free"**
3. Sign up with GitHub (recommended)

### Step 2: Create New Web Service

1. In Render Dashboard, click **"New +"**
2. Select **"Web Service"**
3. Connect your GitHub repository:
   - Click **"Connect GitHub"**
   - Select repository: `Cardano-max/fashioncore_ws`
   - Select branch: `claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM`

### Step 3: Configure Web Service

Fill in these settings:

**Basic Settings**:
- **Name**: `fashioncore-11za` (or your choice)
- **Region**: Choose closest to your users
- **Branch**: `claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM`
- **Root Directory**: Leave blank
- **Runtime**: `Python 3`

**Build Settings**:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python fashioncore_11za.py`

**Plan**:
- Select **"Free"** (750 hours/month)

### Step 4: Set Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**

Add these one by one:

```
ELEVENZA_API_URL = https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN = https://rangshrii.com/
ELEVENZA_AUTH_TOKEN = <your-11za-auth-token>
ELEVENZA_PHONE_NUMBER = 919725791777
KLING_ACCESS_KEY = ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY = pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC
VERIFY_TOKEN = 1122
TEST_MODE = True
PORT = 8080
PYTHON_VERSION = 3.11.0
```

**Important**: Replace `<your-11za-auth-token>` with your actual token!

### Step 5: Deploy

1. Click **"Create Web Service"**
2. Render will:
   - Clone your repository
   - Install dependencies
   - Start your application
   - Give you a URL like: `https://fashioncore-11za.onrender.com`

⏱️ First deployment takes ~3-5 minutes

### Step 6: Get Your App URL

Once deployed, you'll see:
```
https://fashioncore-11za.onrender.com
```

Copy this URL - you'll need it for the webhook!

---

## 🔗 Configure 11za Webhook

1. Go to **11za Dashboard**: https://app.11za.in/
2. Navigate to **Settings** → **Webhooks**
3. Set **Webhook URL**: `https://fashioncore-11za.onrender.com/webhook`
4. Set **Verify Token**: `1122`
5. Enable events:
   - ✅ Incoming messages
   - ✅ Images
   - ✅ Text messages
6. Click **Save**

---

## ✅ Test Your Deployment

### 1. Health Check

```bash
curl https://fashioncore-11za.onrender.com/health
```

**Expected**: `{"status":"ok","time":1234567890}`

### 2. Landing Page

Open in browser:
```
https://fashioncore-11za.onrender.com/
```

**Expected**: Beautiful landing page with Try-On button

### 3. Admin Dashboard

```
https://fashioncore-11za.onrender.com/admin
```

**Expected**: Admin panel showing try-on attempts

### 4. Webhook Test

```bash
curl "https://fashioncore-11za.onrender.com/webhook?hub.mode=subscribe&hub.verify_token=1122&hub.challenge=test123"
```

**Expected**: `test123`

### 5. WhatsApp Flow (End-to-End)

1. Send **"start"** to: +91 9725791777
2. Bot responds: "👋 Welcome to FashionCore..."
3. Upload a person photo
4. Bot: "✨ Creating your outfit..."
5. Wait 3 seconds (TEST_MODE)
6. Receive mock try-on result! ✅

---

## 🎛️ Render Dashboard Features

### View Logs

1. Go to your service in Render
2. Click **"Logs"** tab
3. See real-time logs

### Monitor

- **Metrics** tab: CPU, Memory usage
- **Events** tab: Deploy history
- **Settings** tab: Update env variables

### Auto-Deploy

Every push to your branch automatically deploys! 🚀

To disable:
- Go to **Settings** → **Build & Deploy**
- Toggle **"Auto-Deploy"** off

---

## 💰 Render Free Tier Details

✅ **750 hours/month** = 24/7 uptime
✅ **512 MB RAM** (enough for this app)
✅ **Shared CPU** (sufficient for chatbot)
✅ **Automatic HTTPS**
✅ **Custom domains** (optional)
✅ **No credit card required**

**Limit**: App may spin down after 15 min of inactivity, takes ~30 seconds to wake up on first request.

**To keep always-on**: Upgrade to paid plan ($7/month) or use a free uptime monitor like UptimeRobot.

---

## 🔄 Update Your App

### Option 1: Git Push (Recommended)

```bash
# Make changes
git add .
git commit -m "Update bot"
git push origin claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM

# Render auto-deploys! ✅
```

### Option 2: Manual Deploy

1. Go to Render Dashboard
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## 🐛 Troubleshooting

### Build Failed

**Check**:
1. Logs tab for error details
2. Ensure `requirements.txt` is correct
3. Verify Python version compatibility

**Solution**: Check build command is `pip install -r requirements.txt`

### App Won't Start

**Check**:
1. Start command is `python fashioncore_11za.py`
2. All environment variables are set
3. PORT is set to 8080

**Solution**: Review logs for startup errors

### Webhook Not Working

**Check**:
1. Webhook URL in 11za: `https://your-app.onrender.com/webhook`
2. Verify token matches: `1122`
3. Test with curl command above

**Solution**: Check webhook endpoint returns 200 OK

### 503 Error

**Cause**: App spinning down (free tier)

**Solution**:
- Wait 30 seconds, try again
- Or upgrade to paid plan ($7/month) for always-on

---

## 📊 Monitoring (Optional)

### Keep App Awake

Use **UptimeRobot** (free):

1. Go to https://uptimerobot.com/
2. Add monitor:
   - Type: HTTP(s)
   - URL: `https://your-app.onrender.com/health`
   - Interval: 5 minutes
3. Free tier pings every 5 min = keeps app awake!

---

## 🎯 Production Checklist

Before going live:

- [ ] Deploy to Render successfully
- [ ] Configure 11za webhook
- [ ] Test health endpoint
- [ ] Test landing page
- [ ] Test WhatsApp flow with TEST_MODE=True
- [ ] Verify admin dashboard works
- [ ] Set up UptimeRobot (optional)
- [ ] Test edge cases (random messages, timeouts)
- [ ] Set TEST_MODE=False when ready for real AI
- [ ] Monitor first real try-on

---

## 🆚 Render vs Railway

| Feature | Render Free | Railway Free |
|---------|-------------|--------------|
| Web Services | ✅ Yes | ❌ DB only (some accounts) |
| Hours/Month | 750 | Limited |
| Auto-Deploy | ✅ Yes | ✅ Yes |
| HTTPS | ✅ Yes | ✅ Yes |
| Spin Down | 15 min idle | On low activity |
| Setup | Easy | Easy |

**Winner**: Render ✅ (for this use case)

---

## 📞 Support

### Render Help
- Docs: https://render.com/docs
- Community: https://community.render.com/
- Status: https://status.render.com/

### Your App Help
- Check logs in Render dashboard
- Review `DEPLOYMENT_GUIDE.md`
- See `INTEGRATION_NOTES.md` for API issues

---

## 🎉 You're Live!

Once deployed:
1. ✅ Your bot is live 24/7
2. ✅ Users can try on clothes via WhatsApp
3. ✅ All data logged to database
4. ✅ Admin dashboard accessible
5. ✅ TEST_MODE prevents AI charges

**Share your app URL**: `https://your-app.onrender.com` 🚀

---

**Next**: Test with real users, monitor logs, adjust as needed!

*Deployed with ❤️ on Render.com*
