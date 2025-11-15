#!/bin/bash
# AetherGrid MongoDB Initialization Script

echo "Initializing AetherGrid MongoDB..."

# Wait for MongoDB to be ready
sleep 5

# Initialize replica set (required for change streams)
mongosh --eval "
try {
  rs.initiate({
    _id: 'rs0',
    members: [{ _id: 0, host: 'localhost:27017' }]
  });
  print('✓ Replica set initialized');
} catch (e) {
  print('Replica set already initialized or error: ' + e);
}
"

# Create database and collections
mongosh aethergrid --eval "
// Create conversations collection
db.createCollection('conversations');
db.conversations.createIndex({ 'conversation_id': 1 }, { unique: true });
db.conversations.createIndex({ 'timestamp': -1 });
db.conversations.createIndex({ 'model': 1 });
print('✓ Conversations collection created');

// Create messages collection
db.createCollection('messages');
db.messages.createIndex({ 'message_id': 1 }, { unique: true });
db.messages.createIndex({ 'conversation_id': 1 });
db.messages.createIndex({ 'timestamp': -1 });
db.messages.createIndex({ 'processed': 1 });
print('✓ Messages collection created');

// Create events collection for analytics
db.createCollection('events');
db.events.createIndex({ 'event_type': 1 });
db.events.createIndex({ 'timestamp': -1 });
print('✓ Events collection created');

print('✓ AetherGrid MongoDB initialized successfully!');
"

echo "MongoDB initialization complete!"
