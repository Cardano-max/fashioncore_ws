# 🚀 Deploy to Glitch (Truly Free - No Card Required!)

## Why Glitch?

✅ **100% Free** - No credit card required whatsoever!
✅ **No Payment Walls** - Actually free, not "free with card verification"
✅ **Simple Setup** - Import from GitHub in minutes
✅ **Auto-Deploy** - Updates automatically from GitHub
✅ **HTTPS** - Automatic SSL certificates
✅ **Python Support** - Full Flask app support

⚠️ **Trade-off**: Apps sleep after 5 minutes of inactivity, wake up in ~5 seconds on first request

---

## 📋 Step-by-Step Deployment

### Step 1: Create Glitch Account

1. Go to https://glitch.com/
2. Click **"Sign In"**
3. Sign in with GitHub (recommended)
4. **No credit card required!** ✅

### Step 2: Import from GitHub

1. Click **"New Project"** button
2. Select **"Import from GitHub"**
3. Paste repository URL:
   ```
   https://github.com/Cardano-max/fashioncore_ws
   ```
4. Wait for import (~1-2 minutes)

### Step 3: Configure Environment Variables

Click **".env"** file in Glitch editor and add:

```env
# 11za WhatsApp API
ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=your-11za-auth-token-here
ELEVENZA_PHONE_NUMBER=917405991551

# AI Service Credentials
KLING_ACCESS_KEY=ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY=pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC

# Webhook Configuration
VERIFY_TOKEN=1122

# Test Mode
TEST_MODE=True

# Port (Glitch uses 3000 by default)
PORT=8080
```

**Important**: Replace `your-11za-auth-token-here` with your actual token!

### Step 4: Update for Glitch

Glitch automatically detects `requirements.txt` and `fashioncore_11za.py`.

The app will start automatically once dependencies are installed.

### Step 5: Get Your App URL

Your Glitch URL will be:
```
https://your-project-name.glitch.me
```

You can find this in the **"Share"** button at the top.

### Step 6: Configure 11za Webhook

1. Go to **11za Dashboard**: https://app.11za.in/
2. Navigate to **Settings** → **Webhooks**
3. Set **Webhook URL**: `https://your-project-name.glitch.me/webhook`
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
curl https://your-project-name.glitch.me/health
```

**Expected**: `{"status":"ok","time":1234567890}`

### 2. Landing Page

Open in browser:
```
https://your-project-name.glitch.me/
```

**Expected**: Beautiful landing page with Try-On button

### 3. Admin Dashboard

```
https://your-project-name.glitch.me/admin
```

**Expected**: Admin panel showing try-on attempts

### 4. Webhook Test

```bash
curl "https://your-project-name.glitch.me/webhook?hub.mode=subscribe&hub.verify_token=1122&hub.challenge=test123"
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

## 🎛️ Glitch Features

### View Logs

Click **"Logs"** button in Glitch editor to see real-time logs

### Edit Code

You can edit code directly in Glitch editor or keep syncing from GitHub

### Share Project

Click **"Share"** → Copy project link

### Remix Project

Others can remix (fork) your project if you make it public

---

## ⚠️ Glitch Free Tier Limitations

### Sleep Behavior

**Issue**: App sleeps after 5 minutes of inactivity

**Impact**: First request after sleep takes ~5 seconds to wake up

**Solution**: Use a free uptime monitor (see below)

### Resources

- **1000 hours/month** project uptime (enough for 24/7 with monitor)
- **512 MB RAM** (sufficient for this app)
- **200 MB disk space** (sufficient)
- **4000 requests/hour** (sufficient for chatbot)

### Keeping App Awake

Use **UptimeRobot** (free) to ping your app:

1. Go to https://uptimerobot.com/
2. Sign up (free, no card required)
3. Add monitor:
   - **Type**: HTTP(s)
   - **URL**: `https://your-project-name.glitch.me/health`
   - **Interval**: 5 minutes
4. This keeps your app awake 24/7! ✅

**Note**: Even without uptime monitor, bot still works - just takes 5 seconds to wake on first message after inactivity.

---

## 🔄 Update Your App

### Option 1: Git Push (Recommended)

Glitch can auto-import from GitHub:

1. Make changes locally
2. Commit and push to GitHub
3. In Glitch, click **"Tools"** → **"Import from GitHub"**
4. Confirm import

```bash
git add .
git commit -m "Update bot"
git push origin claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM
```

### Option 2: Edit in Glitch

Edit files directly in Glitch editor - changes take effect immediately

---

## 🐛 Troubleshooting

### Build Failed

**Check**:
1. Glitch logs for error details
2. Ensure `requirements.txt` is correct
3. Check Python version compatibility

**Solution**: View logs, fix errors, Glitch auto-rebuilds

### App Won't Start

**Check**:
1. All environment variables in `.env` are set
2. `PORT=8080` is set
3. Logs for startup errors

**Solution**: Review logs for the exact error

### Webhook Not Working

**Check**:
1. Webhook URL in 11za: `https://your-project-name.glitch.me/webhook`
2. Verify token matches: `1122`
3. Test with curl command above

**Solution**: Check webhook endpoint returns 200 OK

### App Waking Slowly

**Cause**: App sleeping after 5 min inactivity

**Solution**: Set up UptimeRobot monitor (see above)

### Database Not Persisting

**Note**: Glitch resets the filesystem periodically. For production, consider:
- Using external database (like Supabase free tier)
- Accepting that data may be lost on container restart
- For testing purposes, SQLite should work fine

---

## 📊 Monitoring

### View Logs

Click **"Logs"** button in Glitch editor

### Check Database

Access admin dashboard: `https://your-project-name.glitch.me/admin`

### Stats

Click **"Status"** in Glitch to see:
- Uptime
- Request count
- Resource usage

---

## 🎯 Production Checklist

Before going live:

- [ ] Deploy to Glitch successfully
- [ ] Configure 11za webhook
- [ ] Test health endpoint
- [ ] Test landing page
- [ ] Test WhatsApp flow with TEST_MODE=True
- [ ] Verify admin dashboard works
- [ ] Set up UptimeRobot to keep awake (optional)
- [ ] Test edge cases (random messages, timeouts)
- [ ] Verify no "Kling AI" visible anywhere
- [ ] Test auto-reset functionality
- [ ] Set TEST_MODE=False when ready for real AI
- [ ] Monitor first real try-on

---

## 🆚 Platform Comparison

| Feature | Glitch | Koyeb | Render | Railway |
|---------|--------|-------|--------|---------|
| Card Required? | ❌ **No** | ⚠️ Yes | ⚠️ Yes | ⚠️ Yes |
| Actually Free? | ✅ **Yes** | ⚠️ Card needed | ⚠️ Card needed | ⚠️ Card needed |
| Sleep? | ✅ 5 min (5s wake) | ❌ No | ✅ 15 min | ✅ Yes |
| Python? | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Hours/Month | 1000 | Free tier | 750 | Limited |

**Winner**: Glitch! 🏆 (Only truly card-free option)

---

## 💡 Pro Tips

### Custom Domain

Free tier doesn't support custom domains, but `.glitch.me` domain works great!

### Make Project Private

Settings → Make Private (so others can't see your code)

### Boost Project (Optional)

Pay $8/month for:
- No sleep
- More resources
- Custom domain support
- But not required for testing!

### Community

Glitch has a great community - search for Python Flask examples

---

## 📞 Support

### Glitch Help
- Support: https://support.glitch.com/
- Forum: https://support.glitch.com/
- Status: https://status.glitch.com/

### Your App Help
- Check logs in Glitch editor
- Review `DEPLOYMENT_GUIDE.md`
- See `INTEGRATION_NOTES.md` for API issues

---

## 🎉 You're Live!

Once deployed:
1. ✅ Your bot is live (with 5s wake time)
2. ✅ Users can try on clothes via WhatsApp
3. ✅ All data logged to database
4. ✅ Admin dashboard accessible
5. ✅ TEST_MODE prevents AI charges
6. ✅ **No credit card required!** 🎊

**Share your app URL**: `https://your-project-name.glitch.me` 🚀

---

## 🔐 Security Notes

- Keep `.env` file private
- Don't share your project publicly (or remove sensitive data first)
- HTTPS enabled by default
- Environment variables are secure

---

## ⚡ Quick Wake-Up Solution

If the 5-second wake time bothers you:

**Option 1**: Set up UptimeRobot (free, keeps app awake)

**Option 2**: Accept the wake time - most users won't notice

**Option 3**: Upgrade to Glitch Boost ($8/month, no sleep)

**Option 4**: Once you have a card, use Koyeb/Render for no-sleep experience

---

**Next**: Import to Glitch, configure webhook, test your bot! 🎊

*Deployed with ❤️ on Glitch - The actually free platform!*
