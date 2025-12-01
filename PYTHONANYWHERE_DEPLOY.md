# 🚀 Deploy to PythonAnywhere (Free Tier)

## Why PythonAnywhere?

✅ **Free Tier Available** - Beginner account with limitations
✅ **Python/Flask Native** - Built specifically for Python apps
✅ **Always On** - Web apps run 24/7 on free tier
✅ **No Docker Required** - Direct Python deployment
✅ **SSH Access** - Full console access

⚠️ **Limitations on Free Tier**:
- Only whitelisted external sites allowed (API calls restricted)
- May need to whitelist 11za and Kling AI domains
- Limited CPU/bandwidth
- One web app only

---

## ⚠️ Important: API Whitelisting

**Critical Issue**: PythonAnywhere free tier only allows HTTPS requests to whitelisted sites.

Your app needs to call:
- `app.11za.in` (11za WhatsApp API)
- `api.klingai.com` (Kling AI API)

**These are NOT on the whitelist by default**, which means the free tier won't work for this bot without upgrading to a paid plan ($5/month).

---

## 📋 Alternative: Use for Testing Only

You can deploy on PythonAnywhere for **testing the web interface** (landing page, admin dashboard), but WhatsApp/AI features won't work without paid plan.

### Step 1: Create Account

1. Go to https://www.pythonanywhere.com/
2. Click **"Start running Python online"**
3. Sign up with email (no card required for free tier)
4. Activate account via email

### Step 2: Upload Code

**Option A: From GitHub**
```bash
# In PythonAnywhere console
cd ~
git clone https://github.com/Cardano-max/fashioncore_ws.git
cd fashioncore_ws
git checkout claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM
```

**Option B: Upload Files**
- Use "Files" tab
- Upload zip of your repository
- Unzip in console

### Step 3: Install Dependencies

```bash
cd ~/fashioncore_ws
pip3.10 install --user -r requirements.txt
```

### Step 4: Configure Web App

1. Go to **"Web"** tab
2. Click **"Add a new web app"**
3. Choose **"Manual configuration"**
4. Select **Python 3.10**

### Step 5: Set Up WSGI

Edit `/var/www/yourusername_pythonanywhere_com_wsgi.py`:

```python
import sys
import os

# Add your project directory to path
project_home = '/home/yourusername/fashioncore_ws'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Set environment variables
os.environ['ELEVENZA_API_URL'] = 'https://app.11za.in/apis/template/sendTemplate'
os.environ['ELEVENZA_ORIGIN'] = 'https://rangshrii.com/'
os.environ['ELEVENZA_AUTH_TOKEN'] = 'your-token-here'
os.environ['ELEVENZA_PHONE_NUMBER'] = '919725791777'
os.environ['KLING_ACCESS_KEY'] = 'ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP'
os.environ['KLING_SECRET_KEY'] = 'pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC'
os.environ['VERIFY_TOKEN'] = '1122'
os.environ['TEST_MODE'] = 'True'

# Import Flask app
from fashioncore_11za import app as application
```

### Step 6: Reload Web App

Click **"Reload"** button in Web tab

Your app will be at: `https://yourusername.pythonanywhere.com`

---

## ⚠️ Major Limitation

**The webhook won't work on free tier** because:
1. 11za API (`app.11za.in`) is not whitelisted
2. Kling AI API (`api.klingai.com`) is not whitelisted
3. Free tier blocks external HTTPS requests

**Solutions**:
1. **Upgrade to paid plan** ($5/month) - Gets unrestricted internet access
2. **Use different platform** - One that allows external APIs on free tier
3. **Local development with ngrok** - See below

---

## 💡 Better Alternative: Local Development + ngrok

Since most free platforms now require cards or restrict APIs, consider:

### Use ngrok for Testing

```bash
# Install ngrok
brew install ngrok  # Mac
# or download from https://ngrok.com/

# Run your app locally
python fashioncore_11za.py

# In another terminal, expose to internet
ngrok http 8080
```

This gives you a public URL like `https://abc123.ngrok.io` that you can use for 11za webhook!

**Benefits**:
✅ Truly free
✅ No restrictions on API calls
✅ Easy debugging
✅ Full control

**Limitations**:
⚠️ Requires your computer to be on
⚠️ URL changes each time (unless you pay for static URL)

---

## 🆚 Platform Reality Check

| Platform | Card Required? | Free Tier API Access | Status |
|----------|---------------|---------------------|---------|
| **Glitch** | ❌ No | ✅ Yes | ⚠️ Shut down July 2025 |
| **Heroku** | N/A | N/A | ❌ Shut down 2022 |
| **PythonAnywhere** | ❌ No | ❌ **Whitelisted only** | ⚠️ Need paid for APIs |
| **Koyeb** | ⚠️ Yes | ✅ Yes | Requires card |
| **Render** | ⚠️ Yes | ✅ Yes | Requires card |
| **Railway** | ⚠️ Yes | ✅ Yes | Requires card |
| **ngrok (local)** | ❌ No | ✅ Yes | ✅ **Best for testing** |

---

## 🎯 Recommended Path Forward

### For Testing (No Card):

**Option 1: ngrok + Local** ⭐ BEST
```bash
python fashioncore_11za.py
ngrok http 8080
# Use ngrok URL for webhook
```

**Option 2: Request Whitelisting**
- Email PythonAnywhere support
- Ask to whitelist `app.11za.in` and `api.klingai.com`
- May or may not approve for free tier

### For Production (Need Card):

**Option 1: Koyeb** - $0 to start, card required
- Best performance (no sleep)
- Docker support
- Good free tier limits

**Option 2: Render** - $0 to start, card required
- 750 hours/month free
- Sleeps after 15 min
- Easy setup

**Option 3: PythonAnywhere Paid** - $5/month
- Unrestricted API access
- Python-focused
- Good for beginners

---

## 💬 Reality of "Free" Hosting in 2025

Unfortunately, the landscape has changed:

1. **Heroku** shut down free tier (2022)
2. **Glitch** ended project hosting (July 2025)
3. **Most platforms** now require cards to prevent abuse
4. **Free tiers with API access** are nearly extinct

**The Truth**: To run a production bot 24/7 with unrestricted API access, you'll need either:
- A credit card (even if not charged)
- A paid plan ($5-10/month)
- Or run locally with ngrok

---

## 🚀 Quick Start: ngrok Method (RECOMMENDED)

Since ngrok is the only truly free option without restrictions:

```bash
# 1. Install ngrok
# Visit: https://ngrok.com/download
# Sign up (free, no card)
# Follow install instructions

# 2. Run your bot locally
cd /home/user/fashioncore_ws
python fashioncore_11za.py

# 3. In new terminal, expose to web
ngrok http 8080

# 4. Copy the HTTPS URL (e.g., https://abc123.ngrok.io)

# 5. Configure 11za webhook
# URL: https://abc123.ngrok.io/webhook
# Token: 1122

# 6. Test your bot!
# Send "start" to +91 9725791777
```

**Pros**:
- ✅ Truly free
- ✅ No restrictions
- ✅ Easy debugging
- ✅ Full API access

**Cons**:
- ⚠️ Computer must stay on
- ⚠️ URL changes (unless paid plan)
- ⚠️ Not for production

---

## 📞 Support

- PythonAnywhere: https://www.pythonanywhere.com/forums/
- ngrok: https://ngrok.com/docs
- This project: See other deployment guides

---

**Bottom Line**: For testing without a card, use ngrok. For production, you'll need a card-verified platform or paid hosting.
