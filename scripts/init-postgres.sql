-- AetherGrid PostgreSQL Schema Initialization
-- This stores metadata about conversations, models, and analytics

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100),
    total_messages INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0
);

-- Create messages metadata table
CREATE TABLE IF NOT EXISTS message_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id VARCHAR(255) UNIQUE NOT NULL,
    conversation_id VARCHAR(255) REFERENCES conversations(conversation_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR(20) NOT NULL,
    token_count INTEGER,
    has_embeddings BOOLEAN DEFAULT FALSE,
    processing_status VARCHAR(50) DEFAULT 'pending'
);

-- Create models registry table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) UNIQUE NOT NULL,
    provider VARCHAR(50),
    vector_dimensions INTEGER,
    max_context_window INTEGER,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create query logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    filters JSONB,
    results_count INTEGER,
    processing_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    queried_by VARCHAR(100)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON message_metadata(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON message_metadata(created_at);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);

-- Insert default models
INSERT INTO models (model_name, provider, vector_dimensions, max_context_window)
VALUES
    ('claude-sonnet-4-5', 'anthropic', 1536, 200000),
    ('gpt-4', 'openai', 1536, 8192),
    ('gpt-4-turbo', 'openai', 1536, 128000)
ON CONFLICT (model_name) DO NOTHING;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for conversations table
DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (for development)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aether;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aether;

-- Success message
SELECT 'AetherGrid PostgreSQL schema initialized successfully!' as status;
