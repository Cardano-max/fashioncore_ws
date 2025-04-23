from fashioncore import app, db
import os

if __name__ == '__main__':
    # Create directories
    os.makedirs('static', exist_ok=True)
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    # Run app
    app.run(host='0.0.0.0', port=8080)
