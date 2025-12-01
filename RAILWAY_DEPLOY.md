# Railway Deployment Configuration

## Environment Variables to Set in Railway Dashboard

Set the following environment variables in your Railway dashboard:

### Required Variables

```bash
# 11za WhatsApp API
ELEVENZA_API_URL=https://app.11za.in/apis/template/sendTemplate
ELEVENZA_ORIGIN=https://rangshrii.com/
ELEVENZA_AUTH_TOKEN=<your-11za-auth-token>
ELEVENZA_PHONE_NUMBER=919725791777

# Kling AI Credentials
KLING_ACCESS_KEY=ALMrJQFypk3HCYMnkNNfa8NJCB9YPeP
KLING_SECRET_KEY=pNYB39FT3kbGEtaCCM3Qr8PkHHBppdC

# Application URLs (will be set by Railway)
IMAGE_URL=https://your-app.up.railway.app
WEBSITE_URL=https://rangshrii.com/

# Webhook Configuration
VERIFY_TOKEN=1122

# Test Mode (set to False for production)
TEST_MODE=True
```

## Deployment Steps

1. **Connect Repository to Railway**
   - Go to https://railway.app/
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `fashioncore_ws` repository
   - Select branch: `claude/virtual-tryon-chat-flow-01DVtLNUfjUwbaKMEDnrZrQM`

2. **Configure Environment Variables**
   - Go to your project in Railway
   - Click on "Variables" tab
   - Add all the environment variables listed above
   - Update `IMAGE_URL` with your Railway URL once deployed

3. **Configure 11za Webhook**
   - Go to your 11za dashboard
   - Navigate to Settings → Webhooks
   - Set Webhook URL: `https://your-app.up.railway.app/webhook`
   - Set Verify Token: `1122`
   - Enable events: Incoming messages, Images, Text

4. **Test Deployment**
   ```bash
   # Health check
   curl https://your-app.up.railway.app/health

   # Should return: {"status": "ok", "time": <timestamp>}
   ```

5. **Test Landing Page**
   Open: `https://your-app.up.railway.app/`

6. **Test WhatsApp Flow**
   - Send "start" to your WhatsApp number: 919725791777
   - Bot should respond with welcome message
   - Upload a person image
   - Receive mock try-on result (TEST_MODE enabled)

## Railway Configuration Files

- `railway.json` - Deployment configuration
- `Procfile` - Process configuration
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version

## Monitoring

### View Logs
```bash
# In Railway dashboard, go to "Deployments" → "View Logs"
```

### Check Database
Access admin dashboard: `https://your-app.up.railway.app/admin`

## Troubleshooting

### Deployment Issues

1. **Build fails**: Check Railway logs for errors
2. **App crashes**: Verify all environment variables are set
3. **Webhook not working**: Verify webhook URL and token in 11za dashboard
4. **Images not processing**: Check TEST_MODE is set to True

### Common Errors

**Error**: `ModuleNotFoundError: No module named 'cv2'`
**Solution**: Railway will install from requirements.txt automatically

**Error**: `Webhook verification failed`
**Solution**: Ensure VERIFY_TOKEN matches in both Railway and 11za dashboard

**Error**: `11za messages not sending`
**Solution**: Verify ELEVENZA_AUTH_TOKEN is correct

## Free Tier Limits

Railway Free Tier includes:
- ✅ $5 free credit per month
- ✅ Unlimited projects
- ✅ Custom domains
- ✅ Automatic HTTPS
- ✅ GitHub integration

This app should run comfortably within the free tier for testing and small-scale production use.

## Scaling

When ready for production:
- Set `TEST_MODE=False` to enable real Kling AI processing
- Monitor credit usage in both Kling AI and 11za dashboards
- Consider upgrading Railway plan if needed
- Monitor database size (SQLite → PostgreSQL for scale)

## Support

- Railway Docs: https://docs.railway.app/
- Railway Discord: https://discord.gg/railway
- Project Issues: Check GitHub repository issues
