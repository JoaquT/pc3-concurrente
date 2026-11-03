package client;

import java.io.*;
import java.net.Socket;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class StressTestClient {
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 5000;

    public static void main(String[] args) throws Exception {
        int numClientes = args.length > 0 ? Integer.parseInt(args[0]) : 15;
        int txPorCliente = args.length > 1 ? Integer.parseInt(args[1]) : 40;
        int maxDelayMs = args.length > 2 ? Integer.parseInt(args[2]) : 250;
        String serverHost = args.length > 3 ? args[3] : "localhost";
        int serverPort = args.length > 4 ? Integer.parseInt(args[4]) : 5000;

        System.out.println("==========================================================");
        System.out.println("            PRUEBA DE ESTRÉS AL CLÚSTER                   ");
        System.out.println("==========================================================");
        System.out.printf("Destino: %s:%d%n", serverHost, serverPort);
        System.out.printf("Clientes Concurrentes: %d | Tx por Cliente: %d | Total Transacciones: %d%n",
                numClientes, txPorCliente, numClientes * txPorCliente);
        System.out.printf("Delay aleatorio entre transacciones: 0 - %d ms%n", maxDelayMs);

        // Cargar imágenes de prueba en memoria para no saturar el disco
        List<byte[]> testImages = new ArrayList<>();
        File datasetDir = new File("dataset");
        String[] classes = {"gato", "mariposa", "perro"};
        for (String c : classes) {
            File cDir = new File(datasetDir, c);
            if (cDir.exists()) {
                File[] files = cDir.listFiles((d, n) -> n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png"));
                if (files != null && files.length > 0) {
                    for (int i = 0; i < Math.min(3, files.length); i++) {
                        try {
                            testImages.add(Files.readAllBytes(files[i].toPath()));
                        } catch (Exception ignored) {}
                    }
                }
            }
        }

        if (testImages.isEmpty()) {
            System.err.println("Error: No se encontraron imágenes en la carpeta 'dataset/'.");
            return;
        }
        System.out.println("Cargadas " + testImages.size() + " imágenes de prueba en memoria RAM.");

        ExecutorService pool = Executors.newFixedThreadPool(numClientes);
        AtomicInteger exitosas = new AtomicInteger(0);
        AtomicInteger fallidas = new AtomicInteger(0);
        AtomicLong sumaTiemposMs = new AtomicLong(0);
        AtomicLong minTiempoMs = new AtomicLong(Long.MAX_VALUE);
        AtomicLong maxTiempoMs = new AtomicLong(0);

        long globalStart = System.currentTimeMillis();

        for (int i = 1; i <= numClientes; i++) {
            final int clienteId = i;
            pool.execute(() -> {
                Random rand = new Random(clienteId * 1000L);
                for (int t = 1; t <= txPorCliente; t++) {
                    byte[] img = testImages.get(rand.nextInt(testImages.size()));
                    long t0 = System.currentTimeMillis();
                    try (Socket s = new Socket(serverHost, serverPort);
                         DataOutputStream out = new DataOutputStream(new BufferedOutputStream(s.getOutputStream()));
                         DataInputStream in = new DataInputStream(new BufferedInputStream(s.getInputStream()))) {

                        out.writeInt(img.length);
                        out.write(img);
                        out.flush();

                        String response = in.readUTF();
                        long rtt = System.currentTimeMillis() - t0;

                        exitosas.incrementAndGet();
                        sumaTiemposMs.addAndGet(rtt);
                        minTiempoMs.accumulateAndGet(rtt, Math::min);
                        maxTiempoMs.accumulateAndGet(rtt, Math::max);

                        if (t % 10 == 0 || t == 1) {
                            String[] parts = response.split("\\|");
                            String pred = parts.length > 0 ? parts[0] : "OK";
                            System.out.printf("[Cliente %02d | Tx %02d] -> Predicción: %s | RTT: %d ms%n",
                                    clienteId, t, pred, rtt);
                        }

                    } catch (Exception e) {
                        fallidas.incrementAndGet();
                        System.err.printf("[Cliente %02d | Tx %02d] FALLO: %s%n", clienteId, t, e.getMessage());
                    }

                    // Delay aleatorio para regular ráfagas de saturación
                    if (maxDelayMs > 0) {
                        try {
                            Thread.sleep(rand.nextInt(maxDelayMs));
                        } catch (InterruptedException ignored) {}
                    }
                }
            });
        }

        pool.shutdown();
        pool.awaitTermination(5, TimeUnit.MINUTES);

        long globalElapsed = System.currentTimeMillis() - globalStart;
        int total = exitosas.get() + fallidas.get();
        double avgMs = exitosas.get() > 0 ? (double) sumaTiemposMs.get() / exitosas.get() : 0.0;
        double throughput = total > 0 ? (exitosas.get() * 1000.0 / globalElapsed) : 0.0;

        System.out.println("==========================================================");
        System.out.println("          RESUMEN DE PRUEBA DE ESTRÉS Y RENDIMIENTO       ");
        System.out.println("==========================================================");
        System.out.printf("Tiempo Total de Prueba:     %.2f s%n", globalElapsed / 1000.0);
        System.out.printf("Transacciones Exitosas:     %d / %d (%.2f%%)%n", exitosas.get(), total, (100.0 * exitosas.get() / total));
        System.out.printf("Transacciones Fallidas:     %d%n", fallidas.get());
        System.out.printf("Latencia Promedio RTT:      %.2f ms%n", avgMs);
        System.out.printf("Latencia Mínima:            %d ms%n", minTiempoMs.get() == Long.MAX_VALUE ? 0 : minTiempoMs.get());
        System.out.printf("Latencia Máxima:            %d ms%n", maxTiempoMs.get());
        System.out.printf("Throughput del Clúster:     %.2f transacciones/segundo%n", throughput);
        System.out.println("==========================================================");
    }
}