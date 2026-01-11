# 🚀 Koyeb + 11za Setup Guide (Step-by-Step)

## Part 1: Deploy to Koyeb

### Step 1: Create New App

1. Log into **Koyeb Dashboard**: https://app.koyeb.com/
2. Click **"Create App"** button
3. Select **"GitHub"** as deployment method
4. Click **"Connect GitHub Account"** (if not already connected)
5. Authorize Koyeb to access your repositories

### Step 2: Configure Repository

**Repository Settings:**
- **Repository**: Select `Cardano-max/fashioncore_ws`
- **Branch**: `claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM`
- **Builder**: Select `Dockerfile`
- **Dockerfile path**: `Dockerfile` (should auto-detect)

### Step 3: Configure App

**App Configuration:**
- **Name**: `fashioncore-11za` (or choose your own)
- **Region**: Choose closest to your users (e.g., `fra` for Europe, `was` for US East)
- **Instance Type**: Select `Free` (512 MB RAM, 0.1 vCPU)

### Step 4: Set Environment Variables

Click **"Advanced"** → **"Environment Variables"** and add these:

```
ELEVENZA_API_URL = https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN = https://rangshrii.com/
ELEVENZA_AUTH_TOKEN = U2FsdGVkX1/e9ymvz3iAqRt4SA7LgwfStvq6pJdz4WP6yhSMsicFgT7duBMdD9V3q+Qs26KbwdBWtiNeTbqdg8sOO42m2QTejji0oVCKq0Iy81tUHFeqnLqgL285ttgrnk7qY+RRXXaM8taUwCwWVWgIuQxTaoaO4J3/JnxXLoiO8z9TZzNeCuPppwrL+v4A
ELEVENZA_PHONE_NUMBER = 917405991551
KLING_ACCESS_KEY = ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY = pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC
VERIFY_TOKEN = 1122
TEST_MODE = True
PORT = 8080
```

**Important**: Copy these exactly as shown!

### Step 5: Deploy

1. Click **"Create App"** button
2. Koyeb will start building your Docker image
3. Wait 3-5 minutes for first deployment
4. Watch the **"Logs"** tab for build progress

### Step 6: Get Your App URL

Once deployment is complete:
1. Go to **"Settings"** → **"General"**
2. You'll see your app URL like: `https://fashioncore-11za-YOURNAME.koyeb.app`
3. **Copy this URL** - you'll need it for 11za setup!

Example URL: `https://fashioncore-11za-cardano-max.koyeb.app`

---

## Part 2: Configure 11za Webhook

### Step 1: Login to 11za Dashboard

1. Go to: https://app.11za.in/
2. Login with your 11za credentials

### Step 2: Navigate to Webhook Settings

1. Click on **"Settings"** in the left sidebar
2. Click on **"Webhooks"** or **"API Settings"**
3. Look for **"Webhook URL"** section

### Step 3: Configure Webhook

**Enter these details:**

**Webhook URL:**
```
https://fashioncore-11za-YOURNAME.koyeb.app/webhook
```
Replace `YOURNAME` with your actual Koyeb app name!

**Verify Token:**
```
1122
```

**Events to Enable:**
- ✅ **Incoming Messages**
- ✅ **Image Messages**
- ✅ **Text Messages**
- ✅ **Media Messages** (if available)

### Step 4: Save Configuration

1. Click **"Save"** or **"Update Webhook"**
2. 11za might send a verification request to your webhook
3. If successful, you'll see "Webhook verified" or similar message

### Step 5: Test Webhook Connection

In 11za dashboard, look for **"Test Webhook"** button:
1. Click **"Test Webhook"**
2. Should return success (200 OK)
3. If it fails, double-check your Koyeb URL

---

## Part 3: Test Your Bot

### Test 1: Health Check

Open your browser or use curl:
```bash
curl https://fashioncore-11za-YOURNAME.koyeb.app/health
```

**Expected Response:**
```json
{"status":"ok","time":1234567890}
```

### Test 2: Landing Page

Open in browser:
```
https://fashioncore-11za-YOURNAME.koyeb.app/
```

You should see the beautiful landing page with Try-On button!

### Test 3: Admin Dashboard

Open in browser:
```
https://fashioncore-11za-YOURNAME.koyeb.app/admin
```

You should see the admin panel (no attempts yet).

### Test 4: Webhook Verification

```bash
curl "https://fashioncore-11za-YOURNAME.koyeb.app/webhook?hub.mode=subscribe&hub.verify_token=1122&hub.challenge=test123"
```

**Expected Response:**
```
test123
```

### Test 5: WhatsApp Bot (Full Flow!)

**Now the real test:**

1. **Send "start"** to WhatsApp number: **+91 9725791777**

2. **Bot responds** (within 5 seconds):
   ```
   👋 Welcome to FashionCore! Send a full-body photo to begin.
   ```

3. **Upload a person photo** (any photo)

4. **Bot responds**:
   ```
   ✨ Creating your virtual try-on... This will take about 15-20 seconds.
   ```

5. **Wait 3 seconds** (TEST_MODE uses mock response)

6. **Bot sends result**:
   ```
   🎉 Your virtual try-on is ready!
   [Mock result image/URL]
   ```

7. **Bot auto-resets** after 5 seconds ✅

---

## Part 4: Monitor Your Deployment

### View Logs

1. Go to Koyeb Dashboard
2. Click on your app: `fashioncore-11za`
3. Click **"Logs"** tab
4. See real-time logs

**Look for:**
```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
INFO: Webhook handler initialized
INFO: Database initialized
```

### Check Metrics

1. Click **"Metrics"** tab
2. See:
   - CPU usage
   - Memory usage
   - Network traffic
   - Request count

### View Database

Visit admin dashboard:
```
https://fashioncore-11za-YOURNAME.koyeb.app/admin
```

After testing, you should see try-on attempts logged!

---

## 🐛 Troubleshooting

### Issue: Bot Not Responding

**Check:**
1. Koyeb app is running (check dashboard)
2. Webhook URL is correct in 11za
3. Environment variables are set (especially `ELEVENZA_AUTH_TOKEN`)
4. Check Koyeb logs for errors

**Solution:**
```bash
# Test webhook directly
curl -X POST https://fashioncore-11za-YOURNAME.koyeb.app/webhook \
  -H "Content-Type: application/json" \
  -d '{"event":"message","sender":"1234567890","text":"start"}'
```

### Issue: Webhook Failing in 11za

**Check:**
1. URL format: `https://your-app.koyeb.app/webhook` (no trailing slash)
2. Verify token: `1122`
3. Koyeb app is running and healthy

**Solution:**
- Check Koyeb logs when you send test from 11za
- Look for incoming webhook requests

### Issue: Build Failed on Koyeb

**Check Koyeb logs for:**
- Dockerfile syntax errors
- Missing dependencies
- Port configuration issues

**Solution:**
- Check **"Build Logs"** tab
- Look for red error messages
- Redeploy if needed

### Issue: 11za Says "Invalid Token"

**Check:**
1. `ELEVENZA_AUTH_TOKEN` in Koyeb env vars matches your 11za token
2. No extra spaces in the token
3. Token is still valid (hasn't expired)

**Solution:**
- Go to 11za dashboard → API Keys
- Copy token again
- Update in Koyeb → Settings → Environment Variables
- Redeploy app

---

## ✅ Success Checklist

After setup, verify:

- [ ] Koyeb app deployed successfully
- [ ] Health endpoint returns OK
- [ ] Landing page loads
- [ ] Admin dashboard accessible
- [ ] 11za webhook configured
- [ ] Webhook test succeeds in 11za
- [ ] Bot responds to "start" message
- [ ] Bot requests photo
- [ ] Bot processes photo (mock result)
- [ ] Bot auto-resets after showing result
- [ ] Logs show webhook requests
- [ ] Admin dashboard shows attempt

---

## 🎯 Enable Real AI (When Ready)

Once you've tested with TEST_MODE and everything works:

### Step 1: Update Environment Variable

1. Go to Koyeb Dashboard → Your App
2. Click **"Settings"** → **"Environment"**
3. Find `TEST_MODE`
4. Change from `True` to `False`
5. Click **"Save"**
6. App will auto-redeploy

### Step 2: Test with Real AI

1. Send "start" to WhatsApp
2. Upload a person photo
3. Wait **15-20 seconds** (real Kling AI processing)
4. Receive actual try-on result! 🎉

### Step 3: Monitor Usage

- Check Kling AI dashboard for API usage
- Monitor credits consumed
- Check Koyeb logs for processing times

---

## 📊 Production Checklist

Before going fully live:

- [ ] TEST_MODE tested successfully
- [ ] Real AI tested with 1-2 images
- [ ] Edge cases tested (random messages)
- [ ] Auto-reset working (5 seconds)
- [ ] Session timeout working (10 minutes)
- [ ] No "Kling AI" visible in any logs/messages
- [ ] Admin dashboard showing all attempts
- [ ] 11za webhook stable
- [ ] Koyeb metrics looking good
- [ ] Ready to share with users!

---

## 💡 Pro Tips

### Custom Domain (Optional)

1. Koyeb Settings → **"Domains"**
2. Click **"Add Custom Domain"**
3. Enter your domain (e.g., `bot.fashioncore.com`)
4. Update DNS records as shown
5. Update 11za webhook with new domain

### Auto-Deploy

Koyeb auto-deploys on git push:
```bash
git add .
git commit -m "Update bot"
git push origin claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM
# Koyeb auto-deploys! ✅
```

### Scaling

Free tier is sufficient for testing. To scale:
1. Koyeb Settings → **"Scaling"**
2. Upgrade instance type
3. Add more replicas if needed

---

## 🎉 You're Live!

Your FashionCore virtual try-on bot is now:
- ✅ Deployed on Koyeb (24/7 uptime)
- ✅ Connected to 11za WhatsApp
- ✅ Processing virtual try-ons
- ✅ Logging all attempts
- ✅ Auto-resetting after completion
- ✅ Production-ready!

**Share your bot**: Send WhatsApp link to users!

```
https://wa.me/917405991551?text=start
```

Or use your landing page:
```
https://fashioncore-11za-YOURNAME.koyeb.app/
```

---

**Need Help?**
- Check Koyeb logs first
- Review 11za webhook logs
- See INTEGRATION_NOTES.md for API troubleshooting
- Check admin dashboard for database logs
