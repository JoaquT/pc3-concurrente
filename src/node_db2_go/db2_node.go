package main

/*
 * Open AI Cubes - Nodo BD 2 (Go / Fedora + MariaDB)
 * Función: Conectarse a RabbitMQ, consumir eventos de validación en paralelo y
 *          registrarlos de forma segura en MariaDB.
 *
 * RESTRICCIÓN CRÍTICA: Manejo de escrituras concurrentes (1000 transacciones de saturación)
 * mediante doble capa de sincronización explícita:
 *   1) sync.Mutex (dbWriteMutex.Lock / defer dbWriteMutex.Unlock) entre goroutines trabajadoras.
 *   2) Transacción SQL ACID explícita (BeginTx + SELECT ... FOR UPDATE + INSERT + UPDATE + Commit/Rollback).
 */

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	_ "github.com/go-sql-driver/mysql"
	amqp "github.com/rabbitmq/amqp091-go"
)

// Estructuras para deserializar el mensaje JSON proveniente de RabbitMQ
type PredictionDetail struct {
	PredictedClass  string             `json:"predicted_class"`
	Confidence      float64            `json:"confidence"`
	Probabilities   map[string]float64 `json:"probabilities"`
	InferenceTimeMs float64            `json:"inference_time_ms"`
}

type ValidationEventMessage struct {
	RequestID     string           `json:"request_id"`
	UserID        string           `json:"user_id"`
	NodeID        string           `json:"node_id"`
	ExpectedLabel string           `json:"expected_label,omitempty"`
	Result        PredictionDetail `json:"result"`
}

// Mutex explícito a nivel de proceso para sincronizar goroutines concurrentes
var dbWriteMutex sync.Mutex

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}

// Persistencia transaccional protegida por sync.Mutex y bloqueo de fila SQL (FOR UPDATE)
func registerValidationEvent(db *sql.DB, evt ValidationEventMessage) error {
	dbWriteMutex.Lock()
	defer dbWriteMutex.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead})
	if err != nil {
		return fmt.Errorf("error iniciando transacción SQL: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	reqID := evt.RequestID
	if reqID == "" {
		reqID = fmt.Sprintf("val_%d", time.Now().UnixNano())
	}
	userID := evt.UserID
	if userID == "" {
		userID = "anonymous"
	}
	predLabel := evt.Result.PredictedClass
	if predLabel == "" {
		predLabel = "DESCONOCIDO"
	}
	conf := evt.Result.Confidence

	// Regla de validación: una predicción se marca como VERIFIED si su confianza >= 50%
	// (y coincide con ExpectedLabel en caso de haberse provisto etiqueta esperada)
	isValid := 1
	valStatus := "VERIFIED"
	if conf < 50.0 || (evt.ExpectedLabel != "" && evt.ExpectedLabel != predLabel) {
		isValid = 0
		valStatus = "LOW_CONFIDENCE_OR_MISMATCH"
	}

	// 1. Bloqueo pesimista sobre la fila de contadores globales de auditoría (SELECT ... FOR UPDATE)
	var currentProcessed int64
	err = tx.QueryRowContext(
		ctx,
		"SELECT total_processed FROM validation_metrics WHERE metric_key = 'GLOBAL_VALIDATION' FOR UPDATE",
	).Scan(&currentProcessed)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("error adquiriendo bloqueo FOR UPDATE en validation_metrics: %w", err)
	}

	// 2. Insertar el evento de validación individual
	_, err = tx.ExecContext(
		ctx,
		`INSERT IGNORE INTO validation_events
		 (request_id, user_id, predicted_label, expected_label, confidence, is_valid, validation_status, validator_node)
		 VALUES (?, ?, ?, ?, ?, ?, ?, 'Nodo_BD2_Go')`,
		reqID, userID, predLabel, evt.ExpectedLabel, conf, isValid, valStatus,
	)
	if err != nil {
		return fmt.Errorf("error insertando en validation_events: %w", err)
	}

	// 3. Actualizar contadores globales de validación dentro de la misma transacción
	validInc := 0
	flaggedInc := 0
	if isValid == 1 {
		validInc = 1
	} else {
		flaggedInc = 1
	}

	_, err = tx.ExecContext(
		ctx,
		`UPDATE validation_metrics
		 SET total_processed = total_processed + 1,
		     total_valid = total_valid + ?,
		     total_flagged = total_flagged + ?
		 WHERE metric_key = 'GLOBAL_VALIDATION'`,
		validInc, flaggedInc,
	)
	if err != nil {
		return fmt.Errorf("error actualizando validation_metrics: %w", err)
	}

	// 4. Confirmar transacción atómica
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("error al hacer COMMIT en MariaDB: %w", err)
	}

	log.Printf("[Nodo-BD2-MariaDB] Validación registrada (req=%s, clase=%s, conf=%.2f%%, estado=%s, total=%d)",
		reqID, predLabel, conf, valStatus, currentProcessed+1)
	return nil
}

func connectMariaDB(dsn string) *sql.DB {
	for {
		db, err := sql.Open("mysql", dsn)
		if err == nil {
			db.SetMaxOpenConns(25)
			db.SetMaxIdleConns(10)
			db.SetConnMaxLifetime(5 * time.Minute)
			if pingErr := db.Ping(); pingErr == nil {
				log.Println("[Nodo-BD2-MariaDB] Conectado exitosamente a MariaDB.")
				return db
			} else {
				err = pingErr
			}
		}
		log.Printf("[Nodo-BD2-MariaDB] Esperando a MariaDB (%v)... Reintentando en 4s", err)
		time.Sleep(4 * time.Second)
	}
}

func main() {
	rabbitURL := getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
	queueName := getEnv("DB2_QUEUE", "db2_validation_queue")

	dbHost := getEnv("MARIADB_HOST", "localhost")
	dbPort := getEnv("MARIADB_PORT", "3306")
	dbUser := getEnv("MARIADB_USER", "cube_user")
	dbPass := getEnv("MARIADB_PASSWORD", "cube_password")
	dbName := getEnv("MARIADB_DATABASE", "openaicubes_db2")

	dsn := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?parseTime=true&charset=utf8mb4", dbUser, dbPass, dbHost, dbPort, dbName)
	db := connectMariaDB(dsn)
	defer db.Close()

	for {
		log.Printf("[Nodo-BD2-MariaDB] Conectando a RabbitMQ en %s...", rabbitURL)
		conn, err := amqp.Dial(rabbitURL)
		if err != nil {
			log.Printf("[Nodo-BD2-MariaDB] Broker no disponible (%v). Reintentando en 4s...", err)
			time.Sleep(4 * time.Second)
			continue
		}

		ch, err := conn.Channel()
		if err != nil {
			log.Printf("[Nodo-BD2-MariaDB] Error abriendo canal AMQP (%v).", err)
			_ = conn.Close()
			time.Sleep(4 * time.Second)
			continue
		}

		_, err = ch.QueueDeclare(queueName, true, false, false, false, nil)
		if err != nil {
			log.Printf("[Nodo-BD2-MariaDB] Error declarando cola '%s': %v", queueName, err)
			_ = ch.Close()
			_ = conn.Close()
			time.Sleep(4 * time.Second)
			continue
		}

		_ = ch.Qos(16, 0, false)

		msgs, err := ch.Consume(queueName, "go_validator_consumer", false, false, false, false, nil)
		if err != nil {
			log.Printf("[Nodo-BD2-MariaDB] Error iniciando consumo: %v", err)
			_ = ch.Close()
			_ = conn.Close()
			time.Sleep(4 * time.Second)
			continue
		}

		log.Printf("[Nodo-BD2-MariaDB] Escuchando eventos de validación en '%s' con pool de goroutines...", queueName)

		// Pool de 8 goroutines trabajadoras sincronizadas por dbWriteMutex + Transacción SQL
		var wg sync.WaitGroup
		numWorkers := 8
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for d := range msgs {
					var evt ValidationEventMessage
					if err := json.Unmarshal(d.Body, &evt); err != nil {
						log.Printf("[Worker-%d] JSON inválido: %v", workerID, err)
						_ = d.Nack(false, false)
						continue
					}

					if err := registerValidationEvent(db, evt); err != nil {
						log.Printf("[Worker-%d] Error guardando validación: %v", workerID, err)
						_ = d.Nack(false, true)
						continue
					}
					_ = d.Ack(false)
				}
			}(w)
		}

		wg.Wait()
		_ = ch.Close()
		_ = conn.Close()
		log.Println("[Nodo-BD2-MariaDB] Conexión AMQP cerrada. Reconectando en 4s...")
		time.Sleep(4 * time.Second)
	}
}
