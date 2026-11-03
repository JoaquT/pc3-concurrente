package server;

import com.rabbitmq.client.AMQP;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import com.rabbitmq.client.DeliverCallback;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import javax.imageio.ImageIO;

/**
 * Open AI Cubes - Nodo Backend Central (Java / Ubuntu)
 *
 * Cumple con las Restricciones Estrictas:
 * 1) Hilo Principal 1: ServerSocket TCP puro en puerto 5000 para clientes TCP (Chat + Imágenes Base64).
 * 2) Hilo Principal 2: ServerSocket TCP puro en puerto 8080 actuando como Servidor HTTP desde cero (sirve index.html).
 * 3) Comunicación con el Nodo IA exclusivamente vía RabbitMQ (amqp-client) usando colas y correlationId.
 */
public class BackendCentral {

    public static final int TCP_PORT = 5000;
    public static final int HTTP_PORT = 8080;

    public static final String AI_TASKS_QUEUE = "ai_tasks_queue";
    public static final String AI_RESULTS_QUEUE = "ai_results_queue";
    public static final String DB1_METADATA_QUEUE = "db1_metadata_queue";
    public static final String DB2_VALIDATION_QUEUE = "db2_validation_queue";

    private final String rabbitHost;
    private final int rabbitPort;
    private final String rabbitUser;
    private final String rabbitPass;

    private Connection rabbitConnection;
    private Channel rabbitChannel;
    private final Object channelLock = new Object();

    // Mapa concurrente para correlacionar respuestas asíncronas de RabbitMQ con el cliente solicitante
    private final Map<String, CompletableFuture<String>> pendingRequests = new ConcurrentHashMap<>();

    // Lista thread-safe de clientes TCP conectados para difusión de chat
    private final List<PrintWriter> tcpClients = new CopyOnWriteArrayList<>();

    // Historial thread-safe de mensajes de chat para sincronizar clientes TCP (puerto 5000) y clientes Web HTTP (puerto 8080)
    private final List<String> chatHistory = new CopyOnWriteArrayList<>();

    // Pool de hilos para manejo concurrente de conexiones TCP y HTTP
    private final ExecutorService clientPool = Executors.newCachedThreadPool();

    public BackendCentral(String rabbitHost, int rabbitPort, String rabbitUser, String rabbitPass) {
        this.rabbitHost = rabbitHost;
        this.rabbitPort = rabbitPort;
        this.rabbitUser = rabbitUser;
        this.rabbitPass = rabbitPass;
    }

    // =========================================================================
    // 1. CONEXIÓN Y ENRUTAMIENTO CON MIDDLEWARE RABBITMQ (amqp-client)
    // =========================================================================

    public void initRabbitMQ() {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost(this.rabbitHost);
        factory.setPort(this.rabbitPort);
        factory.setUsername(this.rabbitUser);
        factory.setPassword(this.rabbitPass);
        factory.setAutomaticRecoveryEnabled(true);

        while (true) {
            try {
                System.out.println("[Backend-RabbitMQ] Conectando a broker en " + rabbitHost + ":" + rabbitPort + "...");
                this.rabbitConnection = factory.newConnection();
                this.rabbitChannel = this.rabbitConnection.createChannel();

                // Declarar colas hacia el Nodo IA y los Nodos de Bases de Datos (BD1 y BD2)
                this.rabbitChannel.queueDeclare(AI_TASKS_QUEUE, true, false, false, null);
                this.rabbitChannel.queueDeclare(AI_RESULTS_QUEUE, true, false, false, null);
                this.rabbitChannel.queueDeclare(DB1_METADATA_QUEUE, true, false, false, null);
                this.rabbitChannel.queueDeclare(DB2_VALIDATION_QUEUE, true, false, false, null);

                // Escuchar permanentemente las respuestas del Nodo IA en AI_RESULTS_QUEUE
                DeliverCallback deliverCallback = (consumerTag, delivery) -> {
                    String corrId = delivery.getProperties().getCorrelationId();
                    byte[] bodyBytes = delivery.getBody();
                    String responseJson = new String(bodyBytes, StandardCharsets.UTF_8);
                    System.out.println("[Backend-RabbitMQ] Respuesta recibida de Nodo IA (corrId=" + corrId + "): " + responseJson);

                    // Enrutar resultado hacia Nodo BD1 (PostgreSQL) y Nodo BD2 (MariaDB) vía RabbitMQ
                    synchronized (channelLock) {
                        AMQP.BasicProperties dbProps = new AMQP.BasicProperties.Builder()
                                .correlationId(corrId)
                                .contentType("application/json")
                                .deliveryMode(2)
                                .build();
                        rabbitChannel.basicPublish("", DB1_METADATA_QUEUE, dbProps, bodyBytes);
                        rabbitChannel.basicPublish("", DB2_VALIDATION_QUEUE, dbProps, bodyBytes);
                        rabbitChannel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);
                    }

                    if (corrId != null) {
                        CompletableFuture<String> future = pendingRequests.remove(corrId);
                        if (future != null) {
                            future.complete(responseJson);
                        }
                    }
                };

                this.rabbitChannel.basicConsume(AI_RESULTS_QUEUE, false, deliverCallback, consumerTag -> {});
                System.out.println("[Backend-RabbitMQ] Conectado y escuchando cola '" + AI_RESULTS_QUEUE + "'.");
                break;
            } catch (Exception e) {
                System.err.println("[Backend-RabbitMQ] Broker no disponible (" + e.getMessage() + "). Reintentando en 4s...");
                try {
                    Thread.sleep(4000);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    /**
     * Decodifica una imagen en Base64, extrae su matriz 32x32 en escala de grises y su histograma
     * de color de 512 bins, publica la tarea en RabbitMQ y espera de forma asíncrona la respuesta.
     */
    public String routeImageToAINode(String base64Image) throws Exception {
        // Limpiar prefijo data URI si viene desde un navegador (ej. "data:image/png;base64,...")
        String cleanBase64 = base64Image.trim();
        int commaIdx = cleanBase64.indexOf(',');
        if (cleanBase64.startsWith("data:") && commaIdx != -1) {
            cleanBase64 = cleanBase64.substring(commaIdx + 1);
        }

        byte[] imageBytes = Base64.getDecoder().decode(cleanBase64);
        String requestId = UUID.randomUUID().toString();
        String jsonPayload = buildImageTaskJson(requestId, imageBytes);

        CompletableFuture<String> responseFuture = new CompletableFuture<>();
        pendingRequests.put(requestId, responseFuture);

        AMQP.BasicProperties props = new AMQP.BasicProperties.Builder()
                .correlationId(requestId)
                .replyTo(AI_RESULTS_QUEUE)
                .contentType("application/json")
                .deliveryMode(2)
                .build();

        synchronized (channelLock) {
            rabbitChannel.basicPublish(
                    "",
                    AI_TASKS_QUEUE,
                    props,
                    jsonPayload.getBytes(StandardCharsets.UTF_8)
            );
        }
        System.out.println("[Backend-RabbitMQ] Imagen enviada a '" + AI_TASKS_QUEUE + "' con ID: " + requestId);

        try {
            // Esperar respuesta del Nodo IA con timeout de 30 segundos
            return responseFuture.get(30, TimeUnit.SECONDS);
        } catch (Exception e) {
            pendingRequests.remove(requestId);
            throw new IOException("Timeout o error esperando respuesta del Nodo IA: " + e.getMessage());
        }
    }

    /**
     * Convierte los bytes de la imagen decodificada en el JSON de vectores numéricos (32x32 gris + 512 color)
     * requerido por la CNN del Nodo IA en Python.
     */
    private String buildImageTaskJson(String requestId, byte[] imageBytes) throws IOException {
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));
        if (img == null) {
            throw new IOException("No se pudo decodificar la imagen desde los bytes Base64 enviados.");
        }

        int w = img.getWidth();
        int h = img.getHeight();

        // 1. Histograma de color RGB (512 bins)
        int[] bins = new int[512];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int bin = ((r / 32) * 64) + ((g / 32) * 8) + (b / 32);
                bins[bin]++;
            }
        }
        double totalPx = Math.max(1.0, (double) w * h);

        // 2. Redimensionar a 32x32 y extraer luminancia normalizada [0.0, 1.0]
        BufferedImage small = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = small.createGraphics();
        g2.drawImage(img, 0, 0, 32, 32, null);
        g2.dispose();

        StringBuilder sb = new StringBuilder(16384);
        sb.append("{\"request_id\":\"").append(requestId).append("\",\"gray_pixels\":[");
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int rgb = small.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gVal = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                double gray = (0.299 * r + 0.587 * gVal + 0.114 * b) / 255.0;
                if (y > 0 || x > 0) sb.append(',');
                sb.append(String.format(Locale.US, "%.5f", gray));
            }
        }
        sb.append("],\"color_histogram\":[");
        for (int i = 0; i < 512; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.US, "%.6f", bins[i] / totalPx));
        }
        sb.append("]}");
        return sb.toString();
    }

    // =========================================================================
    // 2. HILO PRINCIPAL 1: SERVERSOCKET TCP EN PUERTO 5000 (CLIENTES TCP)
    // =========================================================================

    public void startTcpServerThread() {
        Thread tcpThread = new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(TCP_PORT)) {
                System.out.println("[TCP-Server] Escuchando clientes TCP nativos en el puerto " + TCP_PORT + "...");
                while (!Thread.currentThread().isInterrupted()) {
                    Socket clientSocket = serverSocket.accept();
                    clientPool.submit(() -> handleTcpClient(clientSocket));
                }
            } catch (IOException e) {
                System.err.println("[TCP-Server] Error en ServerSocket " + TCP_PORT + ": " + e.getMessage());
            }
        }, "Thread-TCP-Server-5000");
        tcpThread.start();
    }

    private void handleTcpClient(Socket socket) {
        String clientAddr = socket.getRemoteSocketAddress().toString();
        System.out.println("[TCP-Server] Cliente conectado desde " + clientAddr);

        try (
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true, StandardCharsets.UTF_8)
        ) {
            tcpClients.add(out);
            out.println("WELCOME:Conectado al Backend Central Open AI Cubes (Puerto 5000). Envie 'IMG:<base64>' o '<base64>' para clasificar, o 'CHAT:<mensaje>'.");

            String line;
            while ((line = in.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                if (line.equalsIgnoreCase("QUIT") || line.equalsIgnoreCase("EXIT")) {
                    out.println("BYE");
                    break;
                }

                if (line.startsWith("CHAT:")) {
                    String chatMsg = line.substring(5).trim();
                    broadcastTcp("CHAT:[" + clientAddr + "] " + chatMsg);
                } else {
                    // Soportar prefijo "IMG:" o cadena Base64 directa
                    String base64Data = line.startsWith("IMG:") ? line.substring(4).trim() : line;
                    try {
                        String aiResponse = routeImageToAINode(base64Data);
                        out.println("AI_RESULT:" + aiResponse);
                    } catch (Exception ex) {
                        out.println("AI_ERROR:{\"status\":\"ERROR\",\"message\":\"" + ex.getMessage().replace("\"", "'") + "\"}");
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("[TCP-Server] Cliente desconectado (" + clientAddr + "): " + e.getMessage());
        } finally {
            try {
                socket.close();
            } catch (IOException ignored) {}
        }
    }

    private void broadcastTcp(String message) {
        chatHistory.add(message);
        while (chatHistory.size() > 200) {
            chatHistory.remove(0);
        }
        for (PrintWriter writer : tcpClients) {
            try {
                writer.println(message);
            } catch (Exception ignored) {}
        }
    }

    // =========================================================================
    // 3. HILO PRINCIPAL 2: SERVERSOCKET HTTP EN PUERTO 8080 (DESDE CERO)
    // =========================================================================

    public void startHttpServerThread() {
        Thread httpThread = new Thread(() -> {
            try (ServerSocket httpServer = new ServerSocket(HTTP_PORT)) {
                System.out.println("[HTTP-Server] Servidor HTTP desde cero escuchando en http://localhost:" + HTTP_PORT + " ...");
                while (!Thread.currentThread().isInterrupted()) {
                    Socket httpClient = httpServer.accept();
                    clientPool.submit(() -> handleHttpRequest(httpClient));
                }
            } catch (IOException e) {
                System.err.println("[HTTP-Server] Error en ServerSocket " + HTTP_PORT + ": " + e.getMessage());
            }
        }, "Thread-HTTP-Server-8080");
        httpThread.start();
    }

    private void handleHttpRequest(Socket socket) {
        try (
            BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            OutputStream rawOut = socket.getOutputStream()
        ) {
            String requestLine = reader.readLine();
            if (requestLine == null || requestLine.isEmpty()) return;

            String[] parts = requestLine.split(" ");
            String method = parts.length > 0 ? parts[0] : "GET";
            String path = parts.length > 1 ? parts[1] : "/";

            // Leer cabeceras HTTP
            int contentLength = 0;
            String headerLine;
            while ((headerLine = reader.readLine()) != null && !headerLine.isEmpty()) {
                int colon = headerLine.indexOf(':');
                if (colon != -1) {
                    String key = headerLine.substring(0, colon).trim().toLowerCase(Locale.ROOT);
                    String val = headerLine.substring(colon + 1).trim();
                    if ("content-length".equals(key)) {
                        contentLength = Integer.parseInt(val);
                    }
                }
            }

            // Leer cuerpo si es POST
            String body = "";
            if (contentLength > 0) {
                char[] buf = new char[contentLength];
                int readTotal = 0;
                while (readTotal < contentLength) {
                    int r = reader.read(buf, readTotal, contentLength - readTotal);
                    if (r == -1) break;
                    readTotal += r;
                }
                body = new String(buf, 0, readTotal);
            }

            // Enrutamiento HTTP manual
            if ("OPTIONS".equalsIgnoreCase(method)) {
                sendHttpResponse(rawOut, "204 No Content", "text/plain", new byte[0]);
            } else if ("GET".equalsIgnoreCase(method) && ("/".equals(path) || "/index.html".equals(path))) {
                byte[] htmlBytes = loadIndexHtmlBytes();
                sendHttpResponse(rawOut, "200 OK", "text/html; charset=UTF-8", htmlBytes);
            } else if ("POST".equalsIgnoreCase(method) && "/api/classify".equals(path)) {
                try {
                    String aiJson = routeImageToAINode(body);
                    broadcastTcp("CHAT:[Web-IA] Resultado clasificación -> " + aiJson);
                    sendHttpResponse(rawOut, "200 OK", "application/json; charset=UTF-8", aiJson.getBytes(StandardCharsets.UTF_8));
                } catch (Exception ex) {
                    String errJson = "{\"status\":\"ERROR\",\"error\":\"" + ex.getMessage().replace("\"", "'") + "\"}";
                    sendHttpResponse(rawOut, "500 Internal Server Error", "application/json; charset=UTF-8", errJson.getBytes(StandardCharsets.UTF_8));
                }
            } else if ("POST".equalsIgnoreCase(method) && "/api/chat".equals(path)) {
                String msgText = body.trim();
                if (!msgText.isEmpty()) {
                    broadcastTcp("CHAT:[WebClient] " + msgText);
                }
                sendHttpResponse(rawOut, "200 OK", "application/json; charset=UTF-8", "{\"status\":\"OK\"}".getBytes(StandardCharsets.UTF_8));
            } else if ("GET".equalsIgnoreCase(method) && "/api/chat".equals(path)) {
                StringBuilder jsonChat = new StringBuilder("[");
                for (int i = 0; i < chatHistory.size(); i++) {
                    if (i > 0) jsonChat.append(',');
                    String escaped = chatHistory.get(i).replace("\\", "\\\\").replace("\"", "\\\"");
                    jsonChat.append("\"").append(escaped).append("\"");
                }
                jsonChat.append("]");
                sendHttpResponse(rawOut, "200 OK", "application/json; charset=UTF-8", jsonChat.toString().getBytes(StandardCharsets.UTF_8));
            } else {
                String notFound = "<html><body><h1>404 Not Found</h1></body></html>";
                sendHttpResponse(rawOut, "404 Not Found", "text/html; charset=UTF-8", notFound.getBytes(StandardCharsets.UTF_8));
            }

        } catch (Exception e) {
            System.err.println("[HTTP-Server] Error procesando petición HTTP: " + e.getMessage());
        } finally {
            try {
                socket.close();
            } catch (IOException ignored) {}
        }
    }

    /**
     * Lee el archivo index.html desde el sistema de archivos y devuelve sus bytes crudos.
     */
    private byte[] loadIndexHtmlBytes() throws IOException {
        String[] candidatePaths = {
            "index.html",
            "src/web/index.html",
            "web/index.html"
        };
        for (String p : candidatePaths) {
            File f = new File(p);
            if (f.exists() && f.isFile()) {
                return Files.readAllBytes(f.toPath());
            }
        }
        String fallbackHtml = "<!DOCTYPE html><html lang=\"es\"><head><meta charset=\"UTF-8\">"
                + "<title>Open AI Cubes - Servidor HTTP Nativo</title></head><body>"
                + "<h1>Open AI Cubes - Servidor HTTP Java (ServerSocket 8080) Activo</h1>"
                + "<p>Coloque su archivo <code>index.html</code> en la raíz del proyecto para servirlo.</p>"
                + "</body></html>";
        return fallbackHtml.getBytes(StandardCharsets.UTF_8);
    }

    private void sendHttpResponse(OutputStream out, String status, String contentType, byte[] body) throws IOException {
        String headers = "HTTP/1.1 " + status + "\r\n"
                + "Content-Type: " + contentType + "\r\n"
                + "Content-Length: " + body.length + "\r\n"
                + "Access-Control-Allow-Origin: *\r\n"
                + "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                + "Access-Control-Allow-Headers: Content-Type\r\n"
                + "Connection: close\r\n"
                + "\r\n";
        out.write(headers.getBytes(StandardCharsets.UTF_8));
        out.write(body);
        out.flush();
    }

    // =========================================================================
    // 4. PUNTO DE ENTRADA DEL NODO BACKEND CENTRAL
    // =========================================================================

    public static void main(String[] args) {
        String rabbitHost = System.getenv().getOrDefault("RABBITMQ_HOST", "localhost");
        int rabbitPort = Integer.parseInt(System.getenv().getOrDefault("RABBITMQ_PORT", "5672"));
        String rabbitUser = System.getenv().getOrDefault("RABBITMQ_USER", "guest");
        String rabbitPass = System.getenv().getOrDefault("RABBITMQ_PASS", "guest");

        BackendCentral backend = new BackendCentral(rabbitHost, rabbitPort, rabbitUser, rabbitPass);

        // 1. Inicializar conexión y consumidor de respuestas en RabbitMQ
        backend.initRabbitMQ();

        // 2. Lanzar Hilo Principal 1: ServerSocket TCP en puerto 5000 (Imágenes Base64 + Chat)
        backend.startTcpServerThread();

        // 3. Lanzar Hilo Principal 2: ServerSocket HTTP desde cero en puerto 8080 (index.html)
        backend.startHttpServerThread();
    }
}
