# 11za WhatsApp Integration Notes

## Important Notes for 11za API Integration

### Webhook Format Uncertainty

Since 11za's official documentation is not publicly available, I've implemented support for two common webhook formats:

#### Format 1: Simple messages array
```json
{
  "messages": [
    {
      "from": "919999999999",
      "type": "text",
      "text": "user message"
    }
  ]
}
```

#### Format 2: Meta-style entry format
```json
{
  "entry": [
    {
      "changes": [
        {
          "field": "messages",
          "value": {
            "messages": [
              {
                "from": "919999999999",
                "type": "text",
                "text": {
                  "body": "user message"
                }
              }
            ]
          }
        }
      ]
    }
  ]
}
```

### What to Do If Webhooks Don't Work

1. **Check 11za Dashboard Logs**
   - Look for webhook delivery attempts
   - Check the actual payload format they send

2. **Update the webhook handler**
   - In `fashioncore_11za.py`, locate the `webhook()` function
   - Add logging to see the exact format:
   ```python
   logger.info(f"Raw webhook data: {json.dumps(data, indent=2)}")
   ```

3. **Adjust the message parsing**
   - Update the `handle_message()` function to match the actual format
   - The current implementation handles multiple possible formats

4. **Test with Postman**
   - Use the provided `test_webhook.py` script
   - Send test webhooks to your endpoint
   - Verify the response

### API Endpoint Variations

The provided API endpoint is:
```
https://app.11za.in/apis/template/sendTemplate
```

This endpoint appears to be for **template messages**. You may need different endpoints for:

- **Regular text messages**: Might be `/apis/message/send` or similar
- **Image messages**: Might be `/apis/media/send` or similar
- **Interactive messages**: Check 11za documentation

### Current Implementation Assumptions

1. **Text Messages**: Using the template endpoint with a simple message structure
2. **Image Messages**: Adding `"type": "image"` to the payload
3. **Authentication**: Using Bearer token in Authorization header
4. **Origin Header**: Required based on your configuration

### If Messages Aren't Sending

Try these alternatives in `ElevenzaWhatsAppClient`:

#### Alternative 1: Different endpoint structure
```python
# Instead of /apis/template/sendTemplate
# Try /apis/message/send or /api/v1/messages
```

#### Alternative 2: Different payload structure
```python
# Current structure:
{
    "to": "phone_number",
    "message": "text"
}

# Alternative structure:
{
    "recipient": "phone_number",
    "text": {
        "body": "message"
    }
}
```

#### Alternative 3: Different authentication
```python
# Current: Bearer token in Authorization header
# Alternative: API key in custom header
headers = {
    'X-API-Key': 'your-api-key',
    'Content-Type': 'application/json'
}
```

### Debugging Steps

1. **Enable verbose logging**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test API directly with curl**
   ```bash
   curl -X POST https://app.11za.in/apis/template/sendTemplate \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Origin: https://rangshrii.com/" \
     -d '{
       "to": "919999999999",
       "message": "Test message"
     }'
   ```

3. **Check 11za dashboard**
   - Look for API logs
   - Check sent message history
   - Verify webhook configuration

4. **Contact 11za support**
   - Request official API documentation
   - Ask for webhook payload examples
   - Clarify authentication requirements

### Media/Image Handling

For receiving images via webhook, 11za might provide:

1. **Direct URL** (preferred):
   ```json
   {
     "image": {
       "url": "https://direct-url.com/image.jpg"
     }
   }
   ```

2. **Media ID** (requires additional API call):
   ```json
   {
     "image": {
       "id": "media_id_12345"
     }
   }
   ```

   If using media ID, you'd need:
   ```python
   def download_11za_media(media_id):
       url = f"https://app.11za.in/apis/media/{media_id}"
       response = requests.get(url, headers=self._get_headers())
       # Download and process
   ```

### Kling AI Integration - Already Working

The Kling AI integration is based on official documentation and should work correctly:

- ✅ Authentication via JWT
- ✅ Proper image encoding (base64)
- ✅ Model: kolors-virtual-try-on-v1-5
- ✅ Polling for results
- ✅ Direct URL retrieval

### Production Checklist

Before deploying to production:

- [ ] Verify 11za webhook format with test messages
- [ ] Confirm API endpoint URLs
- [ ] Test message sending with real phone numbers
- [ ] Test image sending with real URLs
- [ ] Verify webhook security (token validation)
- [ ] Set up proper error handling
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerts
- [ ] Test the complete user flow end-to-end
- [ ] Verify Kling AI API quota limits

### Quick Start Testing

1. **Start the application**:
   ```bash
   python fashioncore_11za.py
   ```

2. **In another terminal, run tests**:
   ```bash
   python test_webhook.py
   ```

3. **Check the logs**:
   ```bash
   tail -f app.log
   ```

4. **Test the landing page**:
   Open browser to: `http://localhost:8080`

### Support Resources

- **This codebase**: Fully documented and commented
- **11za Dashboard**: Check for API documentation section
- **Kling AI Docs**: https://www.klingai.com/
- **Railway Logs**: `railway logs --follow`

### Contact for 11za API Details

Since I don't have access to 11za's official documentation, you should:

1. **Check your 11za dashboard** for:
   - API documentation
   - Webhook examples
   - Code samples

2. **Contact 11za support** to get:
   - Official API reference
   - Webhook payload format
   - Authentication requirements
   - Rate limits and quotas

3. **Use their developer portal** if available

Once you have the official documentation, update the `ElevenzaWhatsAppClient` class accordingly.

### Getting Help

If you need to modify the integration:

1. All 11za-related code is in the `ElevenzaWhatsAppClient` class
2. Webhook handling is in the `webhook()` function
3. Message handling logic is in `handle_message()`
4. Logging is comprehensive - check `app.log`

The code is designed to be flexible and easy to modify once you have the exact API specifications.
