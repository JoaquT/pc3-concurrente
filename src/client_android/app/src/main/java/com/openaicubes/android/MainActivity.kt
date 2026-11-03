package com.openaicubes.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Collections
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.ThreadLocalRandom
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Open AI Cubes - Nodo Cliente Android (Kotlin + Sockets TCP Nativos)
 *
 * Cumple con las Restricciones Estrictas:
 * 1) CERO librerías de red de terceros (Sin Retrofit, Volley, OkHttp ni WebSockets).
 *    Usa exclusivamente java.net.Socket del SDK estándar.
 * 2) Concurrencia estricta fuera del Main Thread mediante hilos nativos y ExecutorService,
 *    sincronizando escrituras sobre el socket con un Mutex explícito (socketWriteLock) y
 *    actualizando la interfaz mediante Handler(Looper.getMainLooper()).
 * 3) Protocolo idéntico a DesktopChatClient.java y BackendCentral.java (Puerto TCP 5000):
 *    - Recepción de saludo: "WELCOME:..."
 *    - Envío de texto:      "CHAT:(AndroidUser_xxx) <mensaje>"
 *    - Envío de imagen:     "IMG:<base64_single_line>" (Base64.NO_WRAP)
 *    - Respuesta CNN:       "AI_RESULT:<json>" / "AI_ERROR:<json>"
 * 4) Prueba de Saturación integrada de 1000 transacciones con delays aleatorios (5-45 ms)
 *    calculando Latencia (Mín, Promedio, P95, P99, Máx) y Throughput (img/s).
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TOTAL_SATURATION_REQUESTS = 1000
        private const val SATURATION_WORKERS = 16
    }

    private val username = "AndroidUser_${System.currentTimeMillis() % 1000}"
    private val mainHandler = Handler(Looper.getMainLooper())

    // Pool de hilos para operaciones de E/S puntuales (envío de chat e imágenes de galería)
    private val ioExecutor = Executors.newCachedThreadPool()

    // Socket TCP persistente para la sesión de Chat e Inferencia interactiva
    @Volatile
    private var chatSocket: Socket? = null
    @Volatile
    private var socketOut: PrintWriter? = null
    @Volatile
    private var socketIn: BufferedReader? = null

    // Lock explícito para sincronizar escrituras concurrentes hacia el OutputStream del Socket
    private val socketWriteLock = Any()

    // Estado de la prueba de saturación y última imagen seleccionada en Base64
    private val isSaturationRunning = AtomicBoolean(false)
    @Volatile
    private var lastSelectedBase64: String? = null

    // Referencias de UI
    private lateinit var tvConnectionStatus: TextView
    private lateinit var etServerHost: EditText
    private lateinit var etServerPort: EditText
    private lateinit var btnConnectTcp: Button
    private lateinit var ivSelectedPreview: ImageView
    private lateinit var btnPickAndSendImage: Button
    private lateinit var btnSaturationTest: Button
    private lateinit var pbSaturation: ProgressBar
    private lateinit var tvSaturationMetrics: TextView
    private lateinit var svChatScroll: ScrollView
    private lateinit var tvChatHistory: TextView
    private lateinit var etChatMessage: EditText
    private lateinit var btnSendChat: Button

    // Selector moderno de imágenes de la galería (ActivityResultContracts.GetContent)
    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            processAndSendGalleryImage(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupListeners()
    }

    private fun bindViews() {
        tvConnectionStatus = findViewById(R.id.tvConnectionStatus)
        etServerHost = findViewById(R.id.etServerHost)
        etServerPort = findViewById(R.id.etServerPort)
        btnConnectTcp = findViewById(R.id.btnConnectTcp)
        ivSelectedPreview = findViewById(R.id.ivSelectedPreview)
        btnPickAndSendImage = findViewById(R.id.btnPickAndSendImage)
        btnSaturationTest = findViewById(R.id.btnSaturationTest)
        pbSaturation = findViewById(R.id.pbSaturation)
        tvSaturationMetrics = findViewById(R.id.tvSaturationMetrics)
        svChatScroll = findViewById(R.id.svChatScroll)
        tvChatHistory = findViewById(R.id.tvChatHistory)
        etChatMessage = findViewById(R.id.etChatMessage)
        btnSendChat = findViewById(R.id.btnSendChat)
    }

    private fun setupListeners() {
        btnConnectTcp.setOnClickListener {
            connectToBackendTcp()
        }

        btnSendChat.setOnClickListener {
            sendChatMessage()
        }

        btnPickAndSendImage.setOnClickListener {
            if (socketOut == null) {
                Toast.makeText(this, "Conéctate primero al Backend Central TCP (:5000)", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            pickImageLauncher.launch("image/*")
        }

        btnSaturationTest.setOnClickListener {
            runSaturationTest1000()
        }
    }

    // =========================================================================
    // 1. GESTIÓN DE CONEXIÓN Y LECTURA CONTINUA POR SOCKET TCP PURO (:5000)
    // =========================================================================

    private fun connectToBackendTcp() {
        val host = etServerHost.text.toString().trim()
        if (host.isEmpty()) {
            Toast.makeText(this, "Por favor ingresa manualmente la dirección IP de tu laptop", Toast.LENGTH_SHORT).show()
            etServerHost.requestFocus()
            return
        }
        val port = etServerPort.text.toString().trim().toIntOrNull() ?: 5000

        Thread({
            closeCurrentChatSocket()
            appendChatLog("[Sistema] Conectando por Socket TCP nativo a $host:$port como '$username'...")

            try {
                val sock = Socket()
                sock.connect(InetSocketAddress(host, port), 5000)
                val out = PrintWriter(sock.getOutputStream(), true, StandardCharsets.UTF_8)
                val reader = BufferedReader(InputStreamReader(sock.getInputStream(), StandardCharsets.UTF_8))

                chatSocket = sock
                socketOut = out
                socketIn = reader

                updateStatusUI(connected = true, "$host:$port")

                // Bucle lector bloqueante aislado en este hilo secundario
                while (!Thread.currentThread().isInterrupted) {
                    val serverLine = reader.readLine() ?: break
                    handleIncomingServerLine(serverLine)
                }
            } catch (ex: IOException) {
                appendChatLog("[Error TCP] Conexión finalizada o rechazada ($host:$port): ${ex.message}")
            } finally {
                updateStatusUI(connected = false, "")
                closeCurrentChatSocket()
            }
        }, "Android-TCP-ChatListener").apply {
            isDaemon = true
            start()
        }
    }

    /**
     * Procesa exactamente el protocolo de líneas definido en BackendCentral.java y DesktopChatClient.java.
     */
    private fun handleIncomingServerLine(line: String) {
        when {
            line.startsWith("WELCOME:") -> {
                appendChatLog("[Backend Central] ${line.substring(8)}")
            }
            line.startsWith("CHAT:") -> {
                appendChatLog(line.substring(5))
            }
            line.startsWith("AI_RESULT:") -> {
                val rawJson = line.substring(10)
                val formatted = formatCnnDiagnosis(rawJson)
                appendChatLog(formatted)
            }
            line.startsWith("AI_ERROR:") -> {
                appendChatLog("[Error Clasificación IA] ${line.substring(9)}")
            }
            else -> {
                appendChatLog("[Servidor] $line")
            }
        }
    }

    /**
     * Parsea el JSON devuelto por el Nodo IA a través de RabbitMQ y lo formatea para el chat móvil.
     */
    private fun formatCnnDiagnosis(rawJson: String): String {
        return try {
            val root = JSONObject(rawJson)
            val status = root.optString("status", "OK")
            val nodeId = root.optString("node_id", "Nodo_IA")
            if (status != "OK" || !root.has("result")) {
                return "[Diagnóstico CNN] $rawJson"
            }
            val res = root.getJSONObject("result")
            val predClass = res.optString("predicted_class", "DESCONOCIDO")
            val conf = res.optDouble("confidence", 0.0)
            val timeMs = res.optDouble("inference_time_ms", 0.0)
            val probs = res.optJSONObject("probabilities")

            val pGato = (probs?.optDouble("GATO", 0.0) ?: 0.0) * 100.0
            val pMariposa = (probs?.optDouble("MARIPOSA", 0.0) ?: 0.0) * 100.0
            val pPerro = (probs?.optDouble("PERRO", 0.0) ?: 0.0) * 100.0

            String.format(
                Locale.US,
                "========================================\n" +
                "[DIAGNÓSTICO CNN — %s]\n" +
                "-> Clase Predicha : %s (%.2f%% confianza)\n" +
                "-> Distribución   : GATO %.1f%% | MARIPOSA %.1f%% | PERRO %.1f%%\n" +
                "-> Latencia Nodo  : %.2f ms\n" +
                "========================================",
                nodeId, predClass, conf, pGato, pMariposa, pPerro, timeMs
            )
        } catch (e: Exception) {
            "----------------------------------------\n[Respuesta CNN vía RabbitMQ]\n$rawJson\n----------------------------------------"
        }
    }

    // =========================================================================
    // 2. ENVÍO DE MENSAJES DE CHAT E IMÁGENES EN BASE64 DESDE LA GALERÍA
    // =========================================================================

    private fun sendTcpLineThreadSafe(commandLine: String) {
        synchronized(socketWriteLock) {
            val out = socketOut ?: throw IOException("Socket TCP no conectado")
            out.println(commandLine)
            if (out.checkError()) {
                throw IOException("Fallo al escribir en el OutputStream del Socket TCP")
            }
        }
    }

    private fun sendChatMessage() {
        val msgText = etChatMessage.text.toString().trim()
        if (msgText.isEmpty()) return

        if (socketOut == null) {
            Toast.makeText(this, "No hay conexión TCP activa.", Toast.LENGTH_SHORT).show()
            return
        }

        etChatMessage.setText("")
        ioExecutor.execute {
            try {
                // Mismo protocolo exacto que DesktopChatClient.java: "CHAT:(username) texto"
                sendTcpLineThreadSafe("CHAT:($username) $msgText")
            } catch (ex: IOException) {
                appendChatLog("[Error enviando chat] ${ex.message}")
            }
        }
    }

    private fun processAndSendGalleryImage(uri: Uri) {
        ioExecutor.execute {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                    ?: throw IOException("No se pudo abrir la imagen seleccionada.")
                val originalBitmap = inputStream.use { BitmapFactory.decodeStream(it) }
                    ?: throw IOException("Formato de imagen no válido.")

                // Redimensionar para vista previa y transmisión ágil conservando información cromática y espacial
                val scaledBitmap = Bitmap.createScaledBitmap(originalBitmap, 128, 128, true)

                mainHandler.post {
                    ivSelectedPreview.setImageBitmap(scaledBitmap)
                }

                val baos = ByteArrayOutputStream()
                scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 90, baos)
                val imageBytes = baos.toByteArray()

                // CRÍTICO: Base64.NO_WRAP garantiza una sola línea sin '\n' para readLine() en BackendCentral.java
                val base64Image = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
                lastSelectedBase64 = base64Image

                appendChatLog("[$username] Enviando imagen de galería (${imageBytes.size} bytes -> Base64) por Socket TCP...")

                // Protocolo idéntico a DesktopChatClient.java: aviso CHAT: seguido de comando IMG:<base64>
                sendTcpLineThreadSafe("CHAT:($username) subió una imagen desde Android para clasificación CNN.")
                sendTcpLineThreadSafe("IMG:$base64Image")

            } catch (ex: Exception) {
                appendChatLog("[Error procesando imagen] ${ex.message}")
            }
        }
    }

    // =========================================================================
    // 3. PRUEBA DE SATURACIÓN: 1000 TRANSACCIONES AUTOMÁTICAS CON DELAY ALEATORIO
    // =========================================================================

    private fun runSaturationTest1000() {
        val host = etServerHost.text.toString().trim()
        if (host.isEmpty()) {
            Toast.makeText(this, "Por favor ingresa manualmente la dirección IP de tu laptop primero", Toast.LENGTH_SHORT).show()
            etServerHost.requestFocus()
            return
        }
        if (!isSaturationRunning.compareAndSet(false, true)) {
            Toast.makeText(this, "La prueba de saturación ya está en ejecución.", Toast.LENGTH_SHORT).show()
            return
        }

        val port = etServerPort.text.toString().trim().toIntOrNull() ?: 5000

        btnSaturationTest.isEnabled = false
        pbSaturation.max = TOTAL_SATURATION_REQUESTS
        pbSaturation.progress = 0
        tvSaturationMetrics.text = "Ejecutando prueba de saturación: 0 / $TOTAL_SATURATION_REQUESTS..."

        Thread({
            val payloadBase64 = lastSelectedBase64 ?: generateSyntheticBase64Image()
            appendChatLog(
                "========================================\n" +
                "[PRUEBA DE SATURACIÓN INICIADA]\n" +
                "Destino: $host:$port | Total: $TOTAL_SATURATION_REQUESTS imgs\n" +
                "Hilos Concurrentes: $SATURATION_WORKERS | Delay aleatorio: 5-45 ms\n" +
                "========================================"
            )

            val workerPool = Executors.newFixedThreadPool(SATURATION_WORKERS)
            val completedOk = AtomicInteger(0)
            val completedFail = AtomicInteger(0)
            val latenciesMs = Collections.synchronizedList(ArrayList<Double>(TOTAL_SATURATION_REQUESTS))

            val startNs = System.nanoTime()

            for (i in 1..TOTAL_SATURATION_REQUESTS) {
                workerPool.execute {
                    try {
                        // Delay aleatorio entre 5 ms y 45 ms para simular ráfagas asíncronas
                        val randomDelay = ThreadLocalRandom.current().nextInt(5, 46)
                        Thread.sleep(randomDelay.toLong())

                        val t0 = System.nanoTime()
                        Socket().use { s ->
                            s.connect(InetSocketAddress(host, port), 8000)
                            s.soTimeout = 30000
                            val out = PrintWriter(s.getOutputStream(), true, StandardCharsets.UTF_8)
                            val reader = BufferedReader(InputStreamReader(s.getInputStream(), StandardCharsets.UTF_8))

                            // Consumir saludo inicial WELCOME: del Backend Central
                            reader.readLine()

                            // Enviar comando IMG:<base64> y esperar respuesta AI_RESULT:
                            out.println("IMG:$payloadBase64")
                            val response = reader.readLine()
                            out.println("QUIT")

                            val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
                            if (response != null && response.startsWith("AI_RESULT:")) {
                                latenciesMs.add(elapsedMs)
                                completedOk.incrementAndGet()
                            } else {
                                completedFail.incrementAndGet()
                            }
                        }
                    } catch (_: Exception) {
                        completedFail.incrementAndGet()
                    } finally {
                        val doneTotal = completedOk.get() + completedFail.get()
                        if (doneTotal % 25 == 0 || doneTotal == TOTAL_SATURATION_REQUESTS) {
                            mainHandler.post {
                                pbSaturation.progress = doneTotal
                                tvSaturationMetrics.text = String.format(
                                    Locale.US,
                                    "Progreso: %d/%d (OK: %d | Fallos: %d)",
                                    doneTotal, TOTAL_SATURATION_REQUESTS, completedOk.get(), completedFail.get()
                                )
                            }
                        }
                    }
                }
            }

            workerPool.shutdown()
            while (!workerPool.isTerminated) {
                Thread.sleep(50)
            }

            val totalTimeSec = max(0.001, (System.nanoTime() - startNs) / 1_000_000_000.0)
            val ok = completedOk.get()
            val fail = completedFail.get()
            val throughput = ok / totalTimeSec

            var minMs = 0.0
            var avgMs = 0.0
            var p95Ms = 0.0
            var p99Ms = 0.0
            var maxMs = 0.0

            if (latenciesMs.isNotEmpty()) {
                latenciesMs.sort()
                minMs = latenciesMs.first()
                maxMs = latenciesMs.last()
                avgMs = latenciesMs.average()
                val idx95 = min(latenciesMs.size - 1, floor(latenciesMs.size * 0.95).toInt())
                val idx99 = min(latenciesMs.size - 1, floor(latenciesMs.size * 0.99).toInt())
                p95Ms = latenciesMs[idx95]
                p99Ms = latenciesMs[idx99]
            }

            val summaryReport = String.format(
                Locale.US,
                "========================================\n" +
                "[REPORTE FINAL SATURACIÓN 1000 IMÁGENES]\n" +
                "Exitosas / Total : %d / %d (Fallidas: %d)\n" +
                "Tiempo Total     : %.2f s\n" +
                "Throughput       : %.2f imágenes/seg\n" +
                "Latencia Mínima  : %.2f ms\n" +
                "Latencia Promedio: %.2f ms\n" +
                "Latencia P95     : %.2f ms\n" +
                "Latencia P99     : %.2f ms\n" +
                "Latencia Máxima  : %.2f ms\n" +
                "========================================",
                ok, TOTAL_SATURATION_REQUESTS, fail, totalTimeSec, throughput,
                minMs, avgMs, p95Ms, p99Ms, maxMs
            )

            appendChatLog(summaryReport)

            mainHandler.post {
                tvSaturationMetrics.text = String.format(
                    Locale.US,
                    "Completado: %d/%d OK | %.1f img/s | Prom: %.1f ms | P95: %.1f ms",
                    ok, TOTAL_SATURATION_REQUESTS, throughput, avgMs, p95Ms
                )
                btnSaturationTest.isEnabled = true
                isSaturationRunning.set(false)
            }
        }, "Android-SaturationTest-1000").start()
    }

    /**
     * Genera una imagen sintética 32x32 en memoria codificada en Base64 (sin saltos de línea)
     * en caso de que el usuario ejecute la prueba de saturación antes de elegir una foto de la galería.
     */
    private fun generateSyntheticBase64Image(): String {
        val bmp = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.rgb(180, 110, 60))
        val paint = Paint().apply {
            color = Color.rgb(35, 35, 35)
            style = Paint.Style.STROKE
            strokeWidth = 2f
        }
        canvas.drawCircle(16f, 16f, 10f, paint)

        val baos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.PNG, 100, baos)
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
    }

    // =========================================================================
    // 4. UTILIDADES DE INTERFAZ Y CICLO DE VIDA
    // =========================================================================

    private fun updateStatusUI(connected: Boolean, endpoint: String) {
        mainHandler.post {
            if (connected) {
                tvConnectionStatus.text = "Conectado ($endpoint)"
                tvConnectionStatus.setTextColor(Color.parseColor("#4ADE80"))
            } else {
                tvConnectionStatus.text = "Desconectado"
                tvConnectionStatus.setTextColor(Color.parseColor("#F87171"))
            }
        }
    }

    private fun appendChatLog(message: String) {
        val timeStr = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        mainHandler.post {
            tvChatHistory.append("[$timeStr] $message\n")
            svChatScroll.post {
                svChatScroll.fullScroll(View.FOCUS_DOWN)
            }
        }
    }

    private fun closeCurrentChatSocket() {
        try {
            socketOut?.println("QUIT")
        } catch (_: Exception) {}
        try {
             chatSocket?.close()
        } catch (_: Exception) {}
        chatSocket = null
        socketOut = null
        socketIn = null
    }

    override fun onDestroy() {
        super.onDestroy()
        Thread { closeCurrentChatSocket() }.start()
        ioExecutor.shutdownNow()
    }
}
