# 🚀 Deploy to Koyeb (Truly Free - No Card Required!)

## Why Koyeb?

✅ **Completely Free** - No credit card required!
✅ **Always On** - 512 MB RAM, 0.1 vCPU free forever
✅ **Auto-Deploy** - Connect GitHub, auto-deploy on push
✅ **HTTPS** - Automatic SSL certificates
✅ **Docker Support** - Full control over environment
✅ **No Cold Starts** - Your app stays responsive

---

## 📋 Step-by-Step Deployment

### Step 1: Create Koyeb Account

1. Go to https://www.koyeb.com/
2. Click **"Sign Up"**
3. Sign up with GitHub (recommended)
4. **No credit card required!** ✅

### Step 2: Create New App

1. In Koyeb Dashboard, click **"Create App"**
2. Select **"GitHub"** as deployment method
3. Click **"Connect GitHub Account"**
4. Authorize Koyeb to access your repositories

### Step 3: Configure Deployment

**Repository Settings**:
- **Repository**: Select `Cardano-max/fashioncore_ws`
- **Branch**: `claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM`
- **Builder**: `Docker`
- **Dockerfile**: `Dockerfile` (auto-detected)

**Instance Settings**:
- **Name**: `fashioncore-11za` (or your choice)
- **Region**: Choose closest to your users
- **Instance Type**: `Free` (512 MB RAM, 0.1 vCPU)

### Step 4: Set Environment Variables

Click **"Environment Variables"** and add these:

```bash
# 11za WhatsApp API
ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=<your-11za-auth-token>
ELEVENZA_PHONE_NUMBER=917405991551

# AI Service Credentials
KLING_ACCESS_KEY=ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY=pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC

# Webhook Configuration
VERIFY_TOKEN=1122

# Test Mode (Keep True for testing)
TEST_MODE=True

# Port (Koyeb uses PORT environment variable)
PORT=8080
```

**Important**:
- Replace `<your-11za-auth-token>` with your actual token!
- Keep `TEST_MODE=True` until ready for production

### Step 5: Deploy

1. Click **"Deploy"**
2. Koyeb will:
   - Clone your repository
   - Build Docker image
   - Deploy your application
   - Give you a URL like: `https://fashioncore-11za-yourname.koyeb.app`

⏱️ First deployment takes ~3-5 minutes

### Step 6: Get Your App URL

Once deployed, you'll see:
```
https://fashioncore-11za-yourname.koyeb.app
```

Copy this URL - you'll need it for the webhook!

---

## 🔗 Configure 11za Webhook

1. Go to **11za Dashboard**: https://app.11za.in/
2. Navigate to **Settings** → **Webhooks**
3. Set **Webhook URL**: `https://fashioncore-11za-yourname.koyeb.app/webhook`
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
curl https://fashioncore-11za-yourname.koyeb.app/health
```

**Expected**: `{"status":"ok","time":1234567890}`

### 2. Landing Page

Open in browser:
```
https://fashioncore-11za-yourname.koyeb.app/
```

**Expected**: Beautiful landing page with Try-On button

### 3. Admin Dashboard

```
https://fashioncore-11za-yourname.koyeb.app/admin
```

**Expected**: Admin panel showing try-on attempts

### 4. Webhook Test

```bash
curl "https://fashioncore-11za-yourname.koyeb.app/webhook?hub.mode=subscribe&hub.verify_token=1122&hub.challenge=test123"
```

**Expected**: `test123`

### 5. WhatsApp Flow (End-to-End)

1. Send **"start"** to: +91 9725791777
2. Bot responds: "👋 Welcome to FashionCore..."
3. Upload a person photo
4. Bot: "✨ Creating your outfit..."
5. Wait 3 seconds (TEST_MODE)
6. Receive mock try-on result! ✅
7. Auto-resets after 5 seconds ✅

---

## 🎛️ Koyeb Dashboard Features

### View Logs

1. Go to your app in Koyeb
2. Click **"Logs"** tab
3. See real-time logs

### Monitor

- **Metrics** tab: CPU, Memory, Network usage
- **Deployments** tab: Deploy history
- **Settings** tab: Update env variables, scaling

### Auto-Deploy

Every push to your branch automatically deploys! 🚀

To disable:
- Go to **Settings** → **General**
- Toggle **"Auto-deploy"** off

---

## 💰 Koyeb Free Tier Details

✅ **Free Forever** - No credit card required
✅ **512 MB RAM** (sufficient for this app)
✅ **0.1 vCPU** (shared)
✅ **Automatic HTTPS**
✅ **Custom domains** (optional)
✅ **2.5 GB Docker image storage**
✅ **100 GB bandwidth/month**

**No Sleep**: Unlike Render/Railway, Koyeb free tier doesn't sleep! Your bot stays responsive 24/7 ✅

---

## 🔄 Update Your App

### Option 1: Git Push (Recommended)

```bash
# Make changes
git add .
git commit -m "Update bot"
git push origin claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM

# Koyeb auto-deploys! ✅
```

### Option 2: Manual Deploy

1. Go to Koyeb Dashboard
2. Click **"Deployments"** → **"Redeploy"**

---

## 🐛 Troubleshooting

### Build Failed

**Check**:
1. Logs tab for error details
2. Ensure `Dockerfile` is correct
3. Verify all dependencies in `requirements.txt`

**Solution**: Check build logs for specific error

### App Won't Start

**Check**:
1. All environment variables are set
2. PORT is set to 8080
3. Logs for startup errors

**Solution**: Review logs for the exact error

### Webhook Not Working

**Check**:
1. Webhook URL in 11za: `https://your-app.koyeb.app/webhook`
2. Verify token matches: `1122`
3. Test with curl command above

**Solution**: Check webhook endpoint returns 200 OK

### App Responding Slowly

**Check**:
- Free tier resources may be limited during high traffic
- Check Metrics tab for CPU/Memory usage

**Solution**: Optimize code or upgrade to paid plan if needed

---

## 📊 Monitoring (Optional)

### Custom Domain (Optional)

1. Go to **Settings** → **Domains**
2. Click **"Add Domain"**
3. Follow DNS configuration steps
4. Free on Koyeb! ✅

### Health Monitoring

Koyeb automatically monitors your app health via the `/health` endpoint

---

## 🎯 Production Checklist

Before going live:

- [ ] Deploy to Koyeb successfully
- [ ] Configure 11za webhook
- [ ] Test health endpoint
- [ ] Test landing page
- [ ] Test WhatsApp flow with TEST_MODE=True
- [ ] Verify admin dashboard works
- [ ] Test edge cases (random messages, timeouts)
- [ ] Verify no "Kling AI" visible anywhere
- [ ] Test auto-reset functionality
- [ ] Set TEST_MODE=False when ready for real AI
- [ ] Monitor first real try-on

---

## 🆚 Platform Comparison

| Feature | Koyeb Free | Render Free | Railway Free |
|---------|------------|-------------|--------------|
| Credit Card | ❌ Not Required | ⚠️ Required | ⚠️ Required |
| RAM | 512 MB | 512 MB | 512 MB |
| Sleep/Spin Down | ❌ No | ✅ Yes (15 min) | ✅ Yes |
| Docker Support | ✅ Yes | ⚠️ Limited | ✅ Yes |
| Auto-Deploy | ✅ Yes | ✅ Yes | ✅ Yes |
| Free Forever | ✅ Yes | ⚠️ Hours Limited | ⚠️ Credit Limited |

**Winner**: Koyeb ✅ (No card required + No sleep!)

---

## 📞 Support

### Koyeb Help
- Docs: https://www.koyeb.com/docs
- Discord: https://discord.gg/koyeb
- Status: https://status.koyeb.com/

### Your App Help
- Check logs in Koyeb dashboard
- Review `DEPLOYMENT_GUIDE.md`
- See `INTEGRATION_NOTES.md` for API issues

---

## 🎉 You're Live!

Once deployed:
1. ✅ Your bot is live 24/7 (no sleep!)
2. ✅ Users can try on clothes via WhatsApp
3. ✅ All data logged to database
4. ✅ Admin dashboard accessible
5. ✅ TEST_MODE prevents AI charges
6. ✅ No credit card required!

**Share your app URL**: `https://your-app.koyeb.app` 🚀

---

## 🔐 Security Notes

- Environment variables are encrypted at rest
- HTTPS enabled by default
- Regular security updates via Docker base image
- No exposed credentials in code

---

**Next**: Deploy, test with users, monitor logs, enjoy your free 24/7 bot! 🎊

*Deployed with ❤️ on Koyeb - The truly free cloud platform*
