-- ============================================================================
-- Open AI Cubes - Nodo BD 2 (MariaDB / MySQL)
-- Script de Inicialización: Eventos de Validación y Auditoría Transaccional
-- ============================================================================

CREATE DATABASE IF NOT EXISTS openaicubes_db2;
USE openaicubes_db2;

CREATE TABLE IF NOT EXISTS validation_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id VARCHAR(64) NOT NULL DEFAULT 'anonymous',
    predicted_label VARCHAR(32) NOT NULL,
    expected_label VARCHAR(32) DEFAULT NULL,
    confidence DECIMAL(6, 2) NOT NULL,
    is_valid TINYINT(1) NOT NULL DEFAULT 1,
    validation_status VARCHAR(32) NOT NULL DEFAULT 'VERIFIED',
    validator_node VARCHAR(64) NOT NULL DEFAULT 'Nodo_BD2_Go',
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_val_request_id (request_id),
    INDEX idx_val_status (validation_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS validation_metrics (
    metric_key VARCHAR(32) PRIMARY KEY,
    total_processed BIGINT NOT NULL DEFAULT 0,
    total_valid BIGINT NOT NULL DEFAULT 0,
    total_flagged BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT IGNORE INTO validation_metrics (metric_key, total_processed, total_valid, total_flagged)
VALUES ('GLOBAL_VALIDATION', 0, 0, 0);
