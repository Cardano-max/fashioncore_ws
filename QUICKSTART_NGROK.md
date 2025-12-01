# 🚀 Quick Start: Deploy with ngrok (Truly Free!)

## ✅ What You'll Get

- 🌐 Public HTTPS URL for your local app
- 💬 WhatsApp bot works perfectly
- 🧪 TEST_MODE enabled (no AI charges)
- 🔒 No "Kling AI" visible anywhere
- 📊 Admin dashboard accessible
- 💯 **100% FREE (no card, no restrictions!)**
- 🛠️ Easy debugging (local logs)

**Trade-off**: Your computer must be running

---

## 📋 5-Minute Setup

### 1️⃣ Install ngrok (2 min)

**Mac/Linux:**
```bash
# Mac
brew install ngrok

# Linux
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
  sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list && \
  sudo apt update && sudo apt install ngrok
```

**Windows:**
1. Download from https://ngrok.com/download
2. Extract ngrok.exe
3. Add to PATH (optional)

### 2️⃣ Sign Up & Connect (1 min)

```bash
# Sign up at https://dashboard.ngrok.com/signup (free, no card!)

# Copy your authtoken from dashboard
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### 3️⃣ Run Your Bot (1 min)

```bash
cd /home/user/fashioncore_ws
python fashioncore_11za.py
```

You should see:
```
 * Running on http://0.0.0.0:8080
```

### 4️⃣ Expose with ngrok (30 sec)

**Open new terminal:**
```bash
ngrok http 8080
```

You'll see:
```
Forwarding   https://abc123.ngrok.io -> http://localhost:8080
```

**Copy that HTTPS URL!** (e.g., `https://abc123.ngrok.io`)

### 5️⃣ Configure Webhook (1 min)

```
1. Go to 11za Dashboard → Settings → Webhooks
2. Webhook URL: https://abc123.ngrok.io/webhook
3. Verify Token: 1122
4. Enable: Incoming messages, Images, Text
5. Save
```

### 6️⃣ Test! (30 sec)

```bash
# Health Check
curl https://abc123.ngrok.io/health

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
✅ **100% FREE** - no credit card, no restrictions!
✅ **Easy debugging** - see logs in real-time on your computer

---

## 📱 User Flow

```
User sends "start"
    → Bot: "Send photo" (instant!)
    → User uploads photo
    → Bot: "Creating outfit..."
    → 3 seconds (TEST_MODE)
    → Bot sends mock result
    → Auto-resets after 5 seconds ✅
```

---

## 🎛️ Important URLs

```
Landing Page: https://abc123.ngrok.io/
Admin Dashboard: https://abc123.ngrok.io/admin
Health Check: https://abc123.ngrok.io/health
Webhook: https://abc123.ngrok.io/webhook
Logs: In your terminal (real-time!)
```

---

## 💡 Pro Tips

### Keep Terminal Open
Both terminals must stay open:
- Terminal 1: Running `python fashioncore_11za.py`
- Terminal 2: Running `ngrok http 8080`

### View ngrok Dashboard
Visit http://127.0.0.1:4040 to see:
- All incoming requests
- Request/response details
- Replay requests for debugging

### Custom Subdomain (Optional - Paid)
Free tier gives random URLs that change each restart.

Paid ($8/month) gives:
- Static subdomain (e.g., `https://fashioncore.ngrok.io`)
- Multiple tunnels
- No connection limits

### Production Ready
For 24/7 deployment without your computer:
- Keep computer on 24/7, or
- Use cloud platform (Koyeb/Render with card), or
- Upgrade to ngrok paid + run on server

---

## 🔧 Enable Real AI (When Ready)

```bash
# 1. Stop the bot (Ctrl+C)
# 2. Edit fashioncore_11za.py or set env var:
export TEST_MODE=False
# 3. Restart bot
python fashioncore_11za.py
# 4. ngrok keeps running (no need to restart)
```

---

## 📊 Monitoring

### View Logs
Just watch Terminal 1 - all logs appear in real-time!

### Check Requests
Visit http://127.0.0.1:4040 - ngrok web interface shows:
- Every webhook call
- Request headers
- Response status
- Timing info

### Database
```bash
# Check database stats
sqlite3 tryon_data.db "SELECT COUNT(*) FROM tryon_attempts"

# Or visit admin dashboard
https://abc123.ngrok.io/admin
```

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
- Check both terminals are running
- Verify ngrok URL in 11za webhook
- Check 11za webhook is saved
- Look at Terminal 1 logs

**ngrok URL changed?**
- This happens when you restart ngrok
- Update 11za webhook with new URL
- Or upgrade to paid for static URL

**Connection refused?**
- Make sure bot is running (Terminal 1)
- Check port 8080 is correct
- Try `python fashioncore_11za.py` again

**Webhook failing?**
- Verify URL: `https://your-url.ngrok.io/webhook`
- Verify token: `1122`
- Check ngrok dashboard: http://127.0.0.1:4040

---

## 🆚 Why ngrok?

| Feature | ngrok | Cloud Platforms |
|---------|-------|-----------------|
| **Card?** | ❌ No | ⚠️ Yes (most) |
| **Free?** | ✅ 100% | ⚠️ Restrictions |
| **API Access** | ✅ Unlimited | ⚠️ Limited/Paid |
| **Setup** | ✅ 5 min | ✅ 5-10 min |
| **Debugging** | ✅ Easy (local) | ⚠️ Remote logs |
| **Computer On?** | ⚠️ Required | ✅ Not needed |

**Winner for Testing**: ngrok! 🎊

---

## 📚 Full Documentation

- **ngrok Docs**: https://ngrok.com/docs
- **PYTHONANYWHERE_DEPLOY.md** - Alternative platform info
- **DEPLOYMENT_GUIDE.md** - Full feature documentation
- **SETUP_GUIDE.md** - Technical setup details

---

## ✨ You're All Set!

Your FashionCore virtual try-on bot is:
- ✅ Live via ngrok
- ✅ Working perfectly
- ✅ TEST_MODE enabled
- ✅ Completely free (no card!)
- ✅ White-labeled
- ✅ Easy to debug
- ✅ Production-ready for testing

**Start testing and enjoy!** 🎊

---

## 🔄 Next Time You Want to Run

```bash
# Terminal 1
cd /home/user/fashioncore_ws
python fashioncore_11za.py

# Terminal 2
ngrok http 8080

# Update 11za webhook with new ngrok URL (if changed)
# Test your bot!
```

---

*No credit card. No restrictions. Actually free.* 💯

**Perfect for testing, development, and demos!**
