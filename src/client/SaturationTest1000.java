package client;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Open AI Cubes - Cliente de Prueba de Saturación (1000 Imágenes por Sockets TCP)
 *
 * Envía 1,000 imágenes codificadas en Base64 con delays aleatorios hacia el puerto 5000
 * del Backend Central utilizando exclusivamente Sockets TCP nativos (java.net.Socket).
 * Mide Latencia (Mínima, Promedio, P95, P99, Máxima) y Throughput (transacciones/segundo).
 */
public class SaturationTest1000 {

    public static final int TOTAL_REQUESTS = 1000;
    public static final int CONCURRENT_WORKERS = 20;

    /**
     * Carga una imagen del dataset local si existe, o genera una imagen sintética 32x32 PNG en memoria,
     * devolviendo su representación en Base64 lista para transmitirse por Socket TCP.
     */
    private static String prepareBase64SampleImage() throws Exception {
        File datasetDir = new File("dataset");
        if (datasetDir.exists() && datasetDir.isDirectory()) {
            File[] subDirs = datasetDir.listFiles(File::isDirectory);
            if (subDirs != null) {
                for (File sub : subDirs) {
                    File[] imgs = sub.listFiles((d, name) -> {
                        String low = name.toLowerCase(Locale.ROOT);
                        return low.endsWith(".jpg") || low.endsWith(".jpeg") || low.endsWith(".png");
                    });
                    if (imgs != null && imgs.length > 0) {
                        byte[] bytes = Files.readAllBytes(imgs[0].toPath());
                        System.out.println("[SaturationTest] Usando imagen real del dataset: " + imgs[0].getPath()
                                + " (" + bytes.length + " bytes)");
                        return Base64.getEncoder().encodeToString(bytes);
                    }
                }
            }
        }

        // Imagen sintética 32x32 en caso de no encontrar archivo físico
        BufferedImage synthetic = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = synthetic.createGraphics();
        g.setColor(new Color(180, 110, 60));
        g.fillRect(0, 0, 32, 32);
        g.setColor(new Color(40, 40, 40));
        g.drawOval(6, 6, 20, 20);
        g.dispose();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(synthetic, "png", baos);
        return Base64.getEncoder().encodeToString(baos.toByteArray());
    }

    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 5000;
        int totalRequests = args.length > 2 ? Integer.parseInt(args[2]) : TOTAL_REQUESTS;
        int workers = args.length > 3 ? Integer.parseInt(args[3]) : CONCURRENT_WORKERS;

        String base64Payload = prepareBase64SampleImage();

        System.out.println("======================================================================");
        System.out.println("  OPEN AI CUBES - PRUEBA DE SATURACIÓN POR SOCKETS TCP PUROS (:5000)  ");
        System.out.println("======================================================================");
        System.out.println("Destino         : " + host + ":" + port);
        System.out.println("Total Imágenes  : " + totalRequests);
        System.out.println("Hilos Clientes  : " + workers);
        System.out.println("Delay Aleatorio : 5 ms - 45 ms entre envíos por hilo");
        System.out.println("----------------------------------------------------------------------");

        ExecutorService pool = Executors.newFixedThreadPool(workers);
        CountDownLatch latch = new CountDownLatch(totalRequests);

        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);
        List<Double> latenciesMs = Collections.synchronizedList(new ArrayList<>(totalRequests));

        long globalStartNs = System.nanoTime();

        for (int i = 0; i < totalRequests; i++) {
            final int reqIndex = i + 1;
            pool.submit(() -> {
                try {
                    // Delay aleatorio para simular ráfagas asíncronas realistas
                    int randomDelayMs = ThreadLocalRandom.current().nextInt(5, 46);
                    Thread.sleep(randomDelayMs);

                    long t0 = System.nanoTime();
                    try (
                        Socket socket = new Socket(host, port);
                        PrintWriter out = new PrintWriter(socket.getOutputStream(), true, StandardCharsets.UTF_8);
                        BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8))
                    ) {
                        // Consumir saludo inicial WELCOME del servidor TCP
                        in.readLine();

                        // Enviar imagen en Base64 y esperar resultado del Nodo IA (vía RabbitMQ)
                        out.println("IMG:" + base64Payload);
                        String response = in.readLine();
                        out.println("QUIT");

                        double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;

                        if (response != null && response.startsWith("AI_RESULT:")) {
                            latenciesMs.add(elapsedMs);
                            int done = successCount.incrementAndGet();
                            if (done % 100 == 0 || done == totalRequests) {
                                System.out.printf(Locale.US, "[Progreso] %d / %d completadas (Última latencia: %.2f ms)%n",
                                        done, totalRequests, elapsedMs);
                            }
                        } else {
                            errorCount.incrementAndGet();
                            System.err.println("[Error Req #" + reqIndex + "] Respuesta inesperada: " + response);
                        }
                    }
                } catch (Exception ex) {
                    errorCount.incrementAndGet();
                    System.err.println("[Excepción Req #" + reqIndex + "] " + ex.getMessage());
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await();
        pool.shutdown();

        double totalTimeSec = (System.nanoTime() - globalStartNs) / 1_000_000_000.0;
        int ok = successCount.get();
        int fail = errorCount.get();

        double minMs = 0.0, maxMs = 0.0, avgMs = 0.0, p95Ms = 0.0, p99Ms = 0.0;
        if (!latenciesMs.isEmpty()) {
            Collections.sort(latenciesMs);
            minMs = latenciesMs.get(0);
            maxMs = latenciesMs.get(latenciesMs.size() - 1);
            double sum = 0.0;
            for (double v : latenciesMs) sum += v;
            avgMs = sum / latenciesMs.size();
            p95Ms = latenciesMs.get((int) Math.min(latenciesMs.size() - 1, Math.floor(latenciesMs.size() * 0.95)));
            p99Ms = latenciesMs.get((int) Math.min(latenciesMs.size() - 1, Math.floor(latenciesMs.size() * 0.99)));
        }

        double throughput = ok / Math.max(0.001, totalTimeSec);

        System.out.println("======================================================================");
        System.out.println("               REPORTE FINAL DE SATURACIÓN (1000 TX)                  ");
        System.out.println("======================================================================");
        System.out.printf(Locale.US, "Transacciones Exitosas : %d / %d%n", ok, totalRequests);
        System.out.printf(Locale.US, "Transacciones Fallidas : %d%n", fail);
        System.out.printf(Locale.US, "Tiempo Total           : %.2f segundos%n", totalTimeSec);
        System.out.printf(Locale.US, "Throughput             : %.2f imágenes/segundo%n", throughput);
        System.out.println("----------------------------------------------------------------------");
        System.out.printf(Locale.US, "Latencia Mínima        : %.2f ms%n", minMs);
        System.out.printf(Locale.US, "Latencia Promedio      : %.2f ms%n", avgMs);
        System.out.printf(Locale.US, "Latencia P95           : %.2f ms%n", p95Ms);
        System.out.printf(Locale.US, "Latencia P99           : %.2f ms%n", p99Ms);
        System.out.printf(Locale.US, "Latencia Máxima        : %.2f ms%n", maxMs);
        System.out.println("======================================================================");
    }
}
