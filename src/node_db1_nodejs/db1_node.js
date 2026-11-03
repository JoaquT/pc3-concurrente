#!/usr/bin/env node
/**
 * Open AI Cubes - Nodo BD 1 (Node.js / Debian + PostgreSQL)
 * Función: Consumir resultados de clasificación desde RabbitMQ y persistir metadatos
 *          (etiqueta, probabilidad, fecha e historial de usuario) en PostgreSQL.
 *
 * RESTRICCIÓN CRÍTICA: Control de concurrencia explícito mediante:
 *   1) AsyncMutex en memoria (cola FIFO de promesas) para evitar condiciones de carrera en Node.js.
 *   2) Transacción SQL estricta (BEGIN -> pg_advisory_xact_lock -> INSERT/UPSERT -> COMMIT/ROLLBACK)
 *      para soportar ráfagas de 1000 transacciones concurrentes sin corrupción de datos.
 */

const amqp = require("amqplib");
const { Pool } = require("pg");

const RABBITMQ_URL = process.env.RABBITMQ_URL || "amqp://guest:guest@localhost:5672";
const DB1_QUEUE = process.env.DB1_QUEUE || "db1_metadata_queue";

const pgPool = new Pool({
    host: process.env.POSTGRES_HOST || "localhost",
    port: parseInt(process.env.POSTGRES_PORT || "5433", 10),
    user: process.env.POSTGRES_USER || "cube_user",
    password: process.env.POSTGRES_PASSWORD || "cube_password",
    database: process.env.POSTGRES_DB || "openaicubes_db1",
    max: 20,
    idleTimeoutMillis: 30000
});

// ============================================================================
// 1. MECANISMO DE SINCRONIZACIÓN EXPLÍCITO (MUTEX ASÍNCRONO FIFO)
// ============================================================================

class AsyncMutex {
    constructor() {
        this._locked = false;
        this._waitingQueue = [];
    }

    acquire() {
        return new Promise((resolve) => {
            if (!this._locked) {
                this._locked = true;
                resolve();
            } else {
                this._waitingQueue.push(resolve);
            }
        });
    }

    release() {
        if (this._waitingQueue.length > 0) {
            const nextResolve = this._waitingQueue.shift();
            nextResolve();
        } else {
            this._locked = false;
        }
    }

    async runExclusive(callback) {
        await this.acquire();
        try {
            return await callback();
        } finally {
            this.release();
        }
    }
}

const dbWriteMutex = new AsyncMutex();

// ============================================================================
// 2. ESCRITURA TRANSACCIONAL CONCURRENTE EN POSTGRESQL
// ============================================================================

async function saveClassificationMetadata(payload) {
    const requestId = payload.request_id || `req_${Date.now()}_${Math.random()}`;
    const userId = payload.user_id || "anonymous";
    const nodeId = payload.node_id || "Nodo_IA_Python_1";

    const resultObj = payload.result || {};
    const predictedLabel = resultObj.predicted_class || "DESCONOCIDO";
    const confidence = parseFloat(resultObj.confidence || 0.0);
    const probs = resultObj.probabilities || {};
    const probGato = parseFloat(probs.GATO || 0.0);
    const probMariposa = parseFloat(probs.MARIPOSA || 0.0);
    const probPerro = parseFloat(probs.PERRO || 0.0);
    const inferenceMs = parseFloat(resultObj.inference_time_ms || 0.0);

    // Sección crítica protegida tanto por AsyncMutex de aplicación como por Transacción SQL + Advisory Lock
    return await dbWriteMutex.runExclusive(async () => {
        const client = await pgPool.connect();
        try {
            await client.query("BEGIN");

            // Lock transaccional explícito a nivel de motor PostgreSQL (se libera automáticamente en COMMIT/ROLLBACK)
            await client.query("SELECT pg_advisory_xact_lock(1001)");

            const insertHistorySql = `
                INSERT INTO classification_history (
                    request_id, user_id, node_id, predicted_label, confidence,
                    prob_gato, prob_mariposa, prob_perro, inference_time_ms, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP)
                ON CONFLICT (request_id) DO NOTHING
                RETURNING id;
            `;
            await client.query(insertHistorySql, [
                requestId, userId, nodeId, predictedLabel, confidence,
                probGato, probMariposa, probPerro, inferenceMs
            ]);

            const upsertUserSummarySql = `
                INSERT INTO user_activity_summary (
                    user_id, total_classifications, last_predicted_label, last_activity_at
                )
                VALUES ($1, 1, $2, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE SET
                    total_classifications = user_activity_summary.total_classifications + 1,
                    last_predicted_label = EXCLUDED.last_predicted_label,
                    last_activity_at = CURRENT_TIMESTAMP;
            `;
            await client.query(upsertUserSummarySql, [userId, predictedLabel]);

            await client.query("COMMIT");
            console.log(`[Nodo-BD1-Postgres] Registro guardado (req=${requestId}, user=${userId}, clase=${predictedLabel}, conf=${confidence}%)`);
        } catch (err) {
            await client.query("ROLLBACK");
            console.error(`[Nodo-BD1-Postgres] Error en transacción SQL (ROLLBACK ejecutado):`, err.message);
            throw err;
        } finally {
            client.release();
        }
    });
}

// ============================================================================
// 3. CONSUMIDOR RABBITMQ CON PREFETCH CONTROLADO Y RECONEXIÓN
// ============================================================================

async function startNodeDB1() {
    while (true) {
        try {
            console.log(`[Nodo-BD1-Postgres] Conectando a RabbitMQ en ${RABBITMQ_URL}...`);
            const connection = await amqp.connect(RABBITMQ_URL);
            const channel = await connection.createChannel();

            await channel.assertQueue(DB1_QUEUE, { durable: true });
            // Prefetch permite recibir ráfagas concurrentes que serán sincronizadas por el Mutex + Transacción
            await channel.prefetch(10);

            console.log(`[Nodo-BD1-Postgres] Escuchando resultados de clasificación en cola '${DB1_QUEUE}'...`);

            channel.consume(DB1_QUEUE, async (msg) => {
                if (!msg) return;
                try {
                    const data = JSON.parse(msg.content.toString("utf-8"));
                    await saveClassificationMetadata(data);
                    channel.ack(msg);
                } catch (err) {
                    console.error(`[Nodo-BD1-Postgres] Error procesando mensaje:`, err.message);
                    // Reencolar o descartar según el tipo de fallo
                    channel.nack(msg, false, false);
                }
            }, { noAck: false });

            await new Promise((_, reject) => {
                connection.on("close", () => reject(new Error("Conexión AMQP cerrada")));
                connection.on("error", (err) => reject(err));
            });
        } catch (err) {
            console.error(`[Nodo-BD1-Postgres] Desconexión o espera de broker (${err.message}). Reintentando en 4s...`);
            await new Promise((r) => setTimeout(r, 4000));
        }
    }
}

startNodeDB1();
