# 🚀 Quick Start: Deploy to Glitch in 5 Minutes (No Card!)

## ✅ What You'll Get

- 🌐 Live app at `https://your-app.glitch.me`
- 💬 WhatsApp bot that works 24/7*
- 🧪 TEST_MODE enabled (no AI charges)
- 🔒 No "Kling AI" visible anywhere
- 📊 Admin dashboard at `/admin`
- 💯 **100% FREE (no card required!)**

*Sleeps after 5 min inactivity, wakes in 5 seconds

---

## 📋 5-Minute Setup

### 1️⃣ Create Glitch Account (1 min)

```
1. Go to: https://glitch.com/
2. Click "Sign In"
3. Sign in with GitHub
4. No credit card! ✅
```

### 2️⃣ Import Project (2 min)

```
1. Click "New Project"
2. Select "Import from GitHub"
3. Paste: https://github.com/Cardano-max/fashioncore_ws
4. Wait for import (~1-2 minutes)
5. Project opens automatically
```

### 3️⃣ Set Environment Variables (1 min)

```
Click ".env" file in Glitch editor and paste:

ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=<your-11za-token>
ELEVENZA_PHONE_NUMBER=919725791777
KLING_ACCESS_KEY=ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY=pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC
VERIFY_TOKEN=1122
TEST_MODE=True
PORT=8080
```

Replace `<your-11za-token>` with your actual token!

### 4️⃣ Get Your URL (30 sec)

```
Click "Share" button → Copy live site URL
Example: https://your-project-name.glitch.me
```

### 5️⃣ Configure Webhook (1 min)

```
1. Go to 11za Dashboard → Settings → Webhooks
2. Webhook URL: https://your-project-name.glitch.me/webhook
3. Verify Token: 1122
4. Enable: Incoming messages, Images, Text
5. Save
```

### 6️⃣ Test (30 sec)

```bash
# Health Check
curl https://your-project-name.glitch.me/health

# WhatsApp Test
Send "start" to: +91 9725791777
Upload a photo → Get mock result! ✅
```

---

## 🎯 That's It!

Your bot is now live! 🎉

### What Happens Next?

✅ **Bot responds only to trigger words** (start, hi, hello, etc.)
✅ **Auto-resets after showing result** (5 seconds)
✅ **Session timeout after 10 minutes** of inactivity
✅ **TEST_MODE = no AI charges** until you're ready
✅ **No "Kling AI" visible** - completely white-labeled
✅ **100% FREE** - no credit card ever required!

⚠️ **Note**: App sleeps after 5 minutes of inactivity, wakes in ~5 seconds on first request

---

## 📱 User Flow

```
User sends "start"
    → Bot: "Send photo" (may take 5s if sleeping)
    → User uploads photo
    → Bot: "Creating outfit..."
    → 3 seconds (TEST_MODE)
    → Bot sends mock result
    → Auto-resets after 5 seconds ✅
```

---

## 🎛️ Important URLs

```
Landing Page: https://your-app.glitch.me/
Admin Dashboard: https://your-app.glitch.me/admin
Health Check: https://your-app.glitch.me/health
Webhook: https://your-app.glitch.me/webhook
```

---

## ⚡ Keep App Awake (Optional)

### Use UptimeRobot (Free!)

```
1. Go to: https://uptimerobot.com/
2. Sign up (free, no card)
3. Add monitor:
   - Type: HTTP(s)
   - URL: https://your-app.glitch.me/health
   - Interval: 5 minutes
4. App stays awake 24/7! ✅
```

---

## 🔧 Enable Real AI (When Ready)

```
1. In Glitch editor, open ".env"
2. Change: TEST_MODE=False
3. File auto-saves
4. Test with 1-2 real images
5. Monitor AI usage
```

---

## 📊 Monitor Your App

### View Logs
```
Click "Logs" button in Glitch editor
```

### Check Database
```
Visit: https://your-app.glitch.me/admin
```

### Edit Code
```
Edit directly in Glitch editor
Changes take effect immediately
```

---

## 💡 Pro Tips

**Make Private**:
- Settings → Make Private
- Hides your code from others

**Update from GitHub**:
- Tools → Import from GitHub
- Syncs latest changes

**View Stats**:
- Click "Status" button
- See uptime, requests, resources

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
- App might be sleeping (first request takes 5s)
- Check 11za webhook is configured
- Verify ELEVENZA_AUTH_TOKEN in .env
- Check Glitch logs

**Import failed?**
- Make sure repository URL is correct
- Check Glitch logs for errors
- Try again (sometimes Glitch is busy)

**Webhook failing?**
- Verify URL: `https://your-app.glitch.me/webhook`
- Verify token: `1122`
- Test with curl command

---

## 🆚 Why Glitch?

| Feature | Glitch | Others |
|---------|--------|--------|
| Card? | ❌ **No** | ⚠️ Yes |
| Free? | ✅ **100%** | ⚠️ Need Card |
| Setup | ✅ 5 min | ✅ 5 min |
| Sleep? | ✅ 5 min (5s wake) | Various |

**Winner**: Actually free! 🎊

---

## 📚 Full Documentation

- **GLITCH_DEPLOY.md** - Complete deployment guide
- **DEPLOYMENT_GUIDE.md** - Full feature documentation
- **SETUP_GUIDE.md** - Technical setup details
- **INTEGRATION_NOTES.md** - API troubleshooting

---

## ✨ You're All Set!

Your FashionCore virtual try-on bot is:
- ✅ Live on Glitch
- ✅ Working 24/7*
- ✅ TEST_MODE enabled
- ✅ Completely free (no card!)
- ✅ White-labeled
- ✅ Production-ready

*5 second wake time after 5 min inactivity

**Start testing and enjoy!** 🎊

---

*Questions? Check the full docs or Glitch logs!*

**No credit card. No tricks. Actually free.** 💯
