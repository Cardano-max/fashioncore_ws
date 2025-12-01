# 🚀 Quick Start: Deploy to Koyeb in 5 Minutes (No Card Required!)

## ✅ What You'll Get

- 🌐 Live app at `https://your-app.koyeb.app`
- 💬 WhatsApp bot that works 24/7
- 🧪 TEST_MODE enabled (no AI charges)
- 🔒 No "Kling AI" visible anywhere
- 📊 Admin dashboard at `/admin`
- 💯 Completely FREE (no card required!)
- ⚡ No sleep/wake delays

---

## 📋 5-Minute Setup

### 1️⃣ Create Koyeb Account (1 min)

```
1. Go to: https://www.koyeb.com/
2. Click "Sign Up"
3. Sign up with GitHub
4. No credit card required! ✅
```

### 2️⃣ Deploy App (2 min)

```
1. Click "Create App"
2. Select "GitHub" deployment
3. Connect GitHub account
4. Select repository: Cardano-max/fashioncore_ws
5. Select branch: claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM

6. Configure:
   - Builder: Docker
   - Name: fashioncore-11za
   - Region: Choose closest
   - Instance: Free (512 MB RAM)

7. Add Environment Variables:
   ELEVENZA_API_URL = https://app.11za.in/apis/template/sendTemplate
   ELEVENZA_ORIGIN = https://rangshrii.com/
   ELEVENZA_AUTH_TOKEN = <your-11za-token>
   ELEVENZA_PHONE_NUMBER = 919725791777
   KLING_ACCESS_KEY = ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
   KLING_SECRET_KEY = pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC
   VERIFY_TOKEN = 1122
   TEST_MODE = True
   PORT = 8080

8. Click "Deploy"
```

### 3️⃣ Configure 11za Webhook (1 min)

```
1. Copy your Koyeb URL: https://your-app.koyeb.app
2. Go to 11za Dashboard → Settings → Webhooks
3. Webhook URL: https://your-app.koyeb.app/webhook
4. Verify Token: 1122
5. Enable: Incoming messages, Images, Text
6. Save
```

### 4️⃣ Test (1 min)

```bash
# Health Check
curl https://your-app.koyeb.app/health

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
✅ **No sleep/wake delays** - instant response 24/7!

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
Landing Page: https://your-app.koyeb.app/
Admin Dashboard: https://your-app.koyeb.app/admin
Health Check: https://your-app.koyeb.app/health
Webhook: https://your-app.koyeb.app/webhook
```

---

## 🔧 Enable Real AI (When Ready)

```
1. Go to Koyeb Dashboard
2. Your App → Settings → Environment
3. Change: TEST_MODE = False
4. Click "Save" (auto-redeploys)
5. Test with 1-2 real images
6. Monitor AI usage
```

---

## 📊 Monitor Your App

### View Logs
```
Koyeb Dashboard → Your App → Logs tab
```

### Check Metrics
```
Koyeb Dashboard → Your App → Metrics tab
CPU, Memory, Network usage in real-time
```

### Check Database
```
Visit: https://your-app.koyeb.app/admin
```

---

## 💡 Pro Tips

**Custom Domain** (Optional):
- Settings → Domains → Add Domain
- Follow DNS instructions
- Free on Koyeb! ✅

**Auto-Deploy**:
- Every git push auto-deploys
- Check "Deployments" tab for status

**No Sleep Needed**:
- Unlike Render/Railway, Koyeb doesn't sleep
- Your bot is always responsive! ⚡

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
- Check Koyeb logs for errors

**Build failed?**
- Check Logs tab
- Verify Dockerfile exists
- Check requirements.txt

**Webhook failing?**
- Verify URL: `https://your-app.koyeb.app/webhook`
- Verify token: `1122`
- Test with curl command

---

## 🆚 Why Koyeb?

| Feature | Koyeb | Render | Railway |
|---------|-------|--------|---------|
| Card Required? | ❌ No | ⚠️ Yes | ⚠️ Yes |
| Sleep? | ❌ No | ✅ Yes | ✅ Yes |
| Free Forever? | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| Docker? | ✅ Yes | ⚠️ Limited | ✅ Yes |

**Winner**: Koyeb! 🏆

---

## 📚 Full Documentation

- **KOYEB_DEPLOY.md** - Complete deployment guide
- **DEPLOYMENT_GUIDE.md** - Full feature documentation
- **SETUP_GUIDE.md** - Technical setup details
- **INTEGRATION_NOTES.md** - API troubleshooting

---

## ✨ You're All Set!

Your FashionCore virtual try-on bot is:
- ✅ Live on Koyeb
- ✅ Working 24/7
- ✅ TEST_MODE enabled
- ✅ Completely free (no card!)
- ✅ No sleep delays
- ✅ White-labeled
- ✅ Production-ready

**Start testing and enjoy!** 🎊

---

*Questions? Check the full docs or Koyeb logs!*
