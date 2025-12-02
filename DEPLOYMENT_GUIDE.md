# 🚀 Complete Deployment Guide - FashionCore 11za Virtual Try-On

## ✅ What's Been Implemented

### 1. **Smart Bot Behavior (Edge Cases)**

#### 🎯 Trigger Words Only
Bot **ONLY** responds to these words when idle:
- `start`, `hi`, `hello`, `hey`, `begin`, `tryon`, `try on`, `help`
- **All other messages are ignored** (no spam, no accidental responses)

#### ♻️ Auto-Reset After Result
- Bot sends result image
- Waits 5 seconds
- **Automatically resets** session
- User must send trigger word to start again
- **Session always ends cleanly**

#### ⏱️ Session Timeout
- **10-minute timeout** for inactive users
- Automatic cleanup of stale sessions
- Prevents memory leaks
- Session data cleared completely

#### 🛡️ State-Based Protection
| State | Bot Behavior |
|-------|-------------|
| **IDLE** | Only responds to trigger words, ignores everything else |
| **WAITING_FOR_PERSON** | Reminds to send photo, ignores text messages |
| **PROCESSING** | Tells user to wait, ignores all messages |
| **SHOWING_RESULT** | Shows result, auto-resets after 5 seconds |

#### 🚫 Message Filtering
Bot automatically ignores:
- ✅ Status messages
- ✅ Read receipts
- ✅ Reactions/emojis (unless trigger word)
- ✅ Images in wrong state
- ✅ Non-trigger text in IDLE
- ✅ Empty messages
- ✅ System messages

### 2. **TEST_MODE (No API Charges)**
```python
TEST_MODE = True  # Currently enabled
```

**Benefits**:
- ✅ Zero Kling AI API calls
- ✅ Zero credits consumed
- ✅ 3-second mock processing
- ✅ Returns sample result image
- ✅ Full flow testing without cost

### 3. **Railway Deployment Ready**
- ✅ `Procfile` created
- ✅ `railway.json` configured
- ✅ Environment variables documented
- ✅ Free tier deployment guide
- ✅ One-click deployment ready

---

## 🚀 Deploy to Railway (Free Tier)

### Step 1: Connect to Railway

1. **Go to Railway**: https://railway.app/
2. **Sign in** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose**: `Cardano-max/fashioncore_ws`
6. **Select branch**: `claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM`

### Step 2: Configure Environment Variables

In Railway dashboard, go to **Variables** tab and add:

```bash
# 11za WhatsApp API
ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=<your-11za-auth-token-here>
ELEVENZA_PHONE_NUMBER=917405991551

# Kling AI (Already set in code, but can override)
KLING_ACCESS_KEY=ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY=pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC

# Application URLs (Railway will set IMAGE_URL automatically)
WEBSITE_URL=https://rangshrii.com/

# Webhook Configuration
VERIFY_TOKEN=1122

# Test Mode (Keep True for testing, False for production)
TEST_MODE=True
```

**Important**: Replace `<your-11za-auth-token-here>` with your actual token from the screenshots you provided.

### Step 3: Deploy

1. Railway will **automatically deploy**
2. Wait for build to complete (~2-3 minutes)
3. Note your Railway URL: `https://your-app.up.railway.app`

### Step 4: Configure 11za Webhook

1. **Go to 11za Dashboard**: https://app.11za.in/
2. **Navigate to**: Settings → Webhooks
3. **Set Webhook URL**: `https://your-app.up.railway.app/webhook`
4. **Set Verify Token**: `1122`
5. **Enable Events**:
   - ✅ Incoming messages
   - ✅ Images
   - ✅ Text messages

### Step 5: Test Your Deployment

#### A. Health Check
```bash
curl https://your-app.up.railway.app/health
```
**Expected**: `{"status":"ok","time":1234567890}`

#### B. Landing Page
Open in browser: `https://your-app.up.railway.app/`

**Expected**: Beautiful landing page with Try-On button

#### C. Webhook Verification
```bash
curl "https://your-app.up.railway.app/webhook?hub.mode=subscribe&hub.verify_token=1122&hub.challenge=test123"
```
**Expected**: `test123`

#### D. WhatsApp Flow (End-to-End Test)

1. **Send "start"** to: +91 9725791777
2. **Expected**: "👋 Welcome to FashionCore Magic Try-on! Send a full-body photo to begin."
3. **Upload** a person photo
4. **Expected**: "✨ Creating your outfit... 15-20 seconds..."
5. **Wait** 3 seconds (TEST_MODE)
6. **Expected**: Mock result image sent
7. **Expected**: "Love it? Want to try another outfit? Send 'start' to try again! 😊"
8. **Wait** 5 seconds
9. **Session auto-resets** ✅

---

## 🧪 Testing Scenarios

### ✅ Scenario 1: Trigger Words Only
```
User: "Random message"
Bot: (no response) ✅

User: "hello"
Bot: "👋 Welcome to FashionCore..." ✅
```

### ✅ Scenario 2: State Protection
```
User: "start"
Bot: "Send a full-body photo..."
User: "another message"
Bot: "Please send a full-body photo..." ✅
```

### ✅ Scenario 3: Auto-Reset
```
User: "start"
Bot: "Send photo..."
User: (uploads photo)
Bot: (processing... sends result)
Bot: "Love it? Send 'start' to try again!"
(5 seconds pass)
Bot: (session auto-resets) ✅

User: "hello"
Bot: (starts fresh session) ✅
```

### ✅ Scenario 4: Session Timeout
```
User: "start"
Bot: "Send photo..."
(10 minutes pass with no activity)
User: (uploads photo)
Bot: (session timeout, starts fresh) ✅
```

### ✅ Scenario 5: Wrong State Image
```
User: "start"
Bot: "Send photo..."
User: (uploads photo) ✅ Processed
User: (uploads another photo)
Bot: (ignores - wrong state) ✅
```

---

## 📊 Bot Behavior Flow

```
┌─────────────────────────────────────────────────────┐
│                    IDLE STATE                        │
│  Only responds to: start, hi, hello, hey, etc.     │
│  Ignores: All other messages                        │
└─────────────┬───────────────────────────────────────┘
              │ User sends trigger word
              ▼
┌─────────────────────────────────────────────────────┐
│              WAITING_FOR_PERSON                      │
│  Waiting for: Person image                          │
│  On text: "Please send photo..."                   │
│  On image: Process → PROCESSING                     │
│  Timeout: 10 minutes → IDLE                         │
└─────────────┬───────────────────────────────────────┘
              │ User uploads photo
              ▼
┌─────────────────────────────────────────────────────┐
│                 PROCESSING                           │
│  Kling AI: Generating try-on                        │
│  On message: "Please wait..."                       │
│  Duration: 3s (TEST) / 15-20s (PROD)               │
└─────────────┬───────────────────────────────────────┘
              │ Processing complete
              ▼
┌─────────────────────────────────────────────────────┐
│              SHOWING_RESULT                          │
│  Sends: Result image                                │
│  Sends: "Love it? Send 'start'..."                 │
│  Wait: 5 seconds                                    │
│  Auto-reset → IDLE ✅                               │
└─────────────────────────────────────────────────────┘
```

---

## 🎛️ Configuration Options

### Enable/Disable Test Mode

**Current**: TEST_MODE = True (no API charges)

**To Enable Production**:
```bash
# In Railway dashboard:
TEST_MODE=False
```

Then redeploy or restart the app.

### Adjust Session Timeout

In `fashioncore_11za.py`:
```python
SESSION_TIMEOUT = 600  # 10 minutes (in seconds)
```

### Modify Trigger Words

In `fashioncore_11za.py`:
```python
TRIGGER_WORDS = ['start', 'hi', 'hello', 'hey', 'begin', 'tryon', 'try on', 'help']
```

### Change Auto-Reset Delay

In `fashioncore_11za.py` (line ~691):
```python
time.sleep(5)  # Wait 5 seconds before auto-reset
```

---

## 📈 Monitoring & Logs

### Railway Logs
1. Go to Railway dashboard
2. Click on your project
3. Click "Deployments"
4. Click "View Logs"

### Admin Dashboard
```
URL: https://your-app.up.railway.app/admin
```

Features:
- ✅ View all try-on attempts
- ✅ See phone numbers and timestamps
- ✅ View images and results
- ✅ Export to CSV

### Database Stats
Check from Railway shell:
```bash
sqlite3 tryon_data.db "SELECT COUNT(*) FROM tryon_attempts"
```

---

## 🐛 Troubleshooting

### Issue: Bot not responding
**Check**:
1. Railway logs for errors
2. 11za webhook configuration
3. VERIFY_TOKEN matches
4. Environment variables set correctly

**Solution**: Check Railway logs, verify webhook URL

### Issue: Webhook not receiving messages
**Check**:
1. Webhook URL is correct
2. HTTPS is used (Railway provides this)
3. Verify token matches
4. 11za dashboard shows webhook deliveries

**Solution**: Test webhook endpoint with curl

### Issue: Bot responds to everything
**Check**: Code version - should only respond to trigger words
**Solution**: Pull latest code from branch

### Issue: Session not resetting
**Check**: Logs for "Auto-resetting session" message
**Solution**: Verify auto-reset code is present (time.sleep(5) after result)

### Issue: API charges unexpectedly
**Check**: TEST_MODE environment variable
**Solution**: Ensure TEST_MODE=True in Railway dashboard

---

## 💰 Cost Estimates

### Railway Free Tier
- ✅ $5 free credit/month
- ✅ Enough for testing and small production
- ✅ Automatically sleeps when inactive
- ✅ Wakes up instantly on request

### Kling AI Costs (When TEST_MODE=False)
- Check Kling AI pricing dashboard
- Each try-on = 1 API call
- Monitor usage in Kling AI dashboard

### 11za Costs
- Check 11za pricing for your tier
- Per-message or per-conversation pricing
- Monitor in 11za dashboard

**Recommendation**: Keep TEST_MODE=True until ready for production launch.

---

## 🎯 Production Checklist

Before going live:

- [ ] TEST_MODE=False in Railway
- [ ] Test 2-3 real try-ons to verify quality
- [ ] Monitor Kling AI credit usage
- [ ] Verify 11za message delivery
- [ ] Test all trigger words
- [ ] Test session timeout
- [ ] Test auto-reset functionality
- [ ] Verify webhook receives all messages
- [ ] Check admin dashboard logging
- [ ] Set up monitoring/alerts
- [ ] Document any custom changes
- [ ] Train customer support team

---

## 🎉 You're All Set!

Your FashionCore virtual try-on bot is:
- ✅ Smart (only responds to trigger words)
- ✅ Clean (auto-resets after completion)
- ✅ Safe (TEST_MODE prevents charges)
- ✅ Professional (proper state management)
- ✅ Ready to deploy (Railway configured)

**Next Step**: Deploy to Railway and start testing!

---

## 📞 Support Resources

- **Railway Docs**: https://docs.railway.app/
- **Railway Discord**: https://discord.gg/railway
- **Project Documentation**: See README.md, SETUP_GUIDE.md
- **Testing Guide**: See TESTING_REPORT.md
- **Integration Notes**: See INTEGRATION_NOTES.md

---

**Built with ❤️ for FashionCore by Claude**

*Last Updated: December 1, 2025*
