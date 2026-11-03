-- ============================================================================
-- Open AI Cubes - Nodo BD 1 (PostgreSQL)
-- Script de Inicialización: Metadatos de Clasificación e Historial de Usuario
-- ============================================================================

CREATE TABLE IF NOT EXISTS classification_history (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id VARCHAR(64) NOT NULL DEFAULT 'anonymous',
    node_id VARCHAR(64) NOT NULL,
    predicted_label VARCHAR(32) NOT NULL,
    confidence NUMERIC(6, 2) NOT NULL,
    prob_gato NUMERIC(6, 4) DEFAULT 0.0000,
    prob_mariposa NUMERIC(6, 4) DEFAULT 0.0000,
    prob_perro NUMERIC(6, 4) DEFAULT 0.0000,
    inference_time_ms NUMERIC(10, 2) DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_activity_summary (
    user_id VARCHAR(64) PRIMARY KEY,
    total_classifications INTEGER NOT NULL DEFAULT 0,
    last_predicted_label VARCHAR(32),
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_classification_user_id ON classification_history(user_id);
CREATE INDEX IF NOT EXISTS idx_classification_created_at ON classification_history(created_at);
