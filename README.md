# FashionCore WhatsApp Integration

This application provides a WhatsApp integration for FashionCore's virtual try-on technology.

## Features

- WhatsApp bot for virtual try-on
- Admin dashboard to track user interactions and try-on data
- Database storage for user data and images

## Admin Dashboard

The application now includes an admin dashboard to track all WhatsApp try-on attempts:

- View all users who have tried the service
- See the person and garment images uploaded
- View the try-on results
- Track usage statistics

## Deployment Instructions

### Prerequisites

- Railway account (https://railway.app/)
- PostgreSQL database (can be added in Railway)

### Steps to Deploy

1. Push code to GitHub
2. Create a new project in Railway from GitHub
3. Add a PostgreSQL database plugin to your project
4. Set the following environment variables:
   - `ACCESS_TOKEN`: Your WhatsApp API access token
   - `IMAGE_URL`: Your Railway app URL
   - `PHONE_NUMBER_ID`: Your WhatsApp phone number ID
   - `VERIFY_TOKEN`: Verification token for WhatsApp webhook
   - `WEBSITE_URL`: Your website URL
   - `WHATSAPP_API_VERSION`: WhatsApp API version (e.g., v17.0)
   - `SECRET_KEY`: Random string for Flask secret key
   - `DATABASE_URL`: Will be automatically set by Railway PostgreSQL plugin

5. Deploy the application

### Admin Setup

After deployment:

1. Access the admin setup page at `https://your-railway-app-url/admin/setup` to create the default admin user:
   - Username: admin
   - Password: fashioncore2024

2. Access the admin dashboard at:
   - Admin statistics: `https://your-railway-app-url/admin/stats`
   - Full admin interface: `https://your-railway-app-url/admin/`

## Local Development

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

4. The application will be available at http://localhost:8080