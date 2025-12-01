# 🚀 Quick Start: Deploy to Render in 5 Minutes

## ✅ What You'll Get

- 🌐 Live app at `https://your-app.onrender.com`
- 💬 WhatsApp bot that works 24/7
- 🧪 TEST_MODE enabled (no AI charges)
- 🔒 No "Kling AI" visible anywhere
- 📊 Admin dashboard at `/admin`
- 💯 Completely FREE

---

## 📋 5-Minute Setup

### 1️⃣ Create Render Account (1 min)

```
1. Go to: https://render.com/
2. Click "Get Started for Free"
3. Sign up with GitHub
```

### 2️⃣ Deploy App (2 min)

```
1. Click "New +" → "Web Service"
2. Connect GitHub:
   - Repository: Cardano-max/fashioncore_ws
   - Branch: claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM

3. Configure:
   - Name: fashioncore-11za
   - Runtime: Python 3
   - Build: pip install -r requirements.txt
   - Start: python fashioncore_11za.py
   - Plan: FREE

4. Add Environment Variables:
   ELEVENZA_AUTH_TOKEN = <your-11za-token>
   (All others are pre-configured in render.yaml)

5. Click "Create Web Service"
```

### 3️⃣ Configure 11za Webhook (1 min)

```
1. Copy your Render URL: https://your-app.onrender.com
2. Go to 11za Dashboard → Settings → Webhooks
3. Webhook URL: https://your-app.onrender.com/webhook
4. Verify Token: 1122
5. Enable: Incoming messages, Images, Text
6. Save
```

### 4️⃣ Test (1 min)

```bash
# Health Check
curl https://your-app.onrender.com/health

# WhatsApp Test
Send "start" to: +91 9725791777
Upload a photo → Get mock result! ✅
```

---

## 🎯 That's It!

Your bot is now live and working! 🎉

### What Happens Next?

✅ **Bot responds only to trigger words** (start, hi, hello, etc.)
✅ **Auto-resets after showing result** (5 seconds)
✅ **Session timeout after 10 minutes** of inactivity
✅ **TEST_MODE = no AI charges** until you're ready
✅ **No "Kling AI" visible** - completely white-labeled

---

## 📱 User Flow

```
User sends "start"
    → Bot: "Send photo"
    → User uploads photo
    → Bot: "Creating outfit..."
    → 3 seconds (TEST_MODE)
    → Bot sends mock result
    → Auto-resets after 5 seconds ✅
```

---

## 🎛️ Important URLs

```
Landing Page: https://your-app.onrender.com/
Admin Dashboard: https://your-app.onrender.com/admin
Health Check: https://your-app.onrender.com/health
Webhook: https://your-app.onrender.com/webhook
```

---

## 🔧 Enable Real AI (When Ready)

```
1. Go to Render Dashboard
2. Environment Variables
3. Change: TEST_MODE = False
4. Save (auto-redeploys)
5. Test with 1-2 real images
6. Monitor AI usage
```

---

## 📊 Monitor Your App

### View Logs
```
Render Dashboard → Your Service → Logs tab
```

### Check Database
```
Render Dashboard → Your Service → Admin
Visit: https://your-app.onrender.com/admin
```

---

## 💡 Pro Tips

**Keep App Awake** (Optional):
- Use UptimeRobot.com (free)
- Ping `/health` every 5 minutes
- Prevents 15-min spin-down

**Auto-Deploy**:
- Every git push auto-deploys
- Check "Events" tab for deploy status

**Custom Domain** (Optional):
- Settings → Custom Domain
- Add your domain (free on Render)

---

## ⚡ Edge Cases Handled

✅ Bot ignores random messages (IDLE state)
✅ Only responds to trigger words
✅ Auto-resets after result (5s)
✅ Session timeout (10 min)
✅ Image validation
✅ Error recovery
✅ No "Kling AI" in logs
✅ Complete white-label

---

## 🐛 Quick Troubleshooting

**Bot not responding?**
- Check 11za webhook is configured
- Verify ELEVENZA_AUTH_TOKEN is set
- Check Render logs for errors

**503 Error?**
- App spinning down (free tier)
- Wait 30 seconds, try again
- Or set up UptimeRobot

**Webhook failing?**
- Verify URL: `https://your-app.onrender.com/webhook`
- Verify token: `1122`
- Test with curl command

---

## 📚 Full Documentation

- **RENDER_DEPLOY.md** - Complete deployment guide
- **DEPLOYMENT_GUIDE.md** - Full feature documentation
- **SETUP_GUIDE.md** - Technical setup details
- **INTEGRATION_NOTES.md** - API troubleshooting

---

## ✨ You're All Set!

Your FashionCore virtual try-on bot is:
- ✅ Live on Render
- ✅ Working 24/7
- ✅ TEST_MODE enabled
- ✅ Completely free
- ✅ White-labeled
- ✅ Production-ready

**Start testing and enjoy!** 🎊

---

*Questions? Check the full docs or Render logs!*
