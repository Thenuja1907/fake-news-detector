import sys
sys.path.insert(0, 'backend')
from database import user_collection
from werkzeug.security import generate_password_hash

users = [
    {'username': 'demo_user', 'email': 'user@demo.com', 'password': generate_password_hash('anypass'), 'is_admin': False},
    {'username': 'manivannanthenuja', 'email': 'manivannanthenuja@gmail.com', 'password': generate_password_hash('anypass'), 'is_admin': True},
]

for u in users:
    existing = user_collection.find_one({'email': u['email']})
    if existing:
        user_collection.update_one({'email': u['email']}, {'$set': {'password': u['password'], 'is_admin': u['is_admin']}})
        print('Updated: ' + u['email'])
    else:
        user_collection.insert_one(u)
        print('Created: ' + u['email'])

print('Done.')
